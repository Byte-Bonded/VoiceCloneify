"""
Extract HuBERT features and create k-means centroids for StreamVC content encoder.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import librosa
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import HubertModel


class AudioDataset(Dataset):
    """Simple dataset for loading audio files."""
    
    def __init__(self, audio_dir: Path, sample_rate: int = 16000):
        self.audio_files = list(audio_dir.rglob("*.wav")) + list(audio_dir.rglob("*.flac"))
        self.sample_rate = sample_rate
        
        print(f"Found {len(self.audio_files)} audio files")
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        try:
            # Load audio with librosa (more robust)
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Convert to tensor
            waveform = torch.from_numpy(waveform).float()
            
            return waveform
        except Exception as e:
            # If loading fails, return silence
            print(f"Warning: Failed to load {audio_path}: {e}")
            return torch.zeros(self.sample_rate)  # 1 second of silence


def collate_fn(batch):
    """Collate function that handles variable length audio."""
    # Find max length
    max_len = max(x.shape[0] for x in batch)
    
    # Pad all sequences
    padded = []
    for x in batch:
        if x.shape[0] < max_len:
            x = torch.nn.functional.pad(x, (0, max_len - x.shape[0]))
        padded.append(x)
    
    return torch.stack(padded)


@torch.no_grad()
def extract_hubert_features(
    audio_dir: Path,
    model_name: str = "facebook/hubert-base-ls960",
    layer: int = 7,
    batch_size: int = 4,
    device: str = "cuda",
    max_samples: int = None
):
    """Extract HuBERT features from audio files."""
    
    print(f"Loading HuBERT model: {model_name}")
    model = HubertModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    if device == "cuda":
        model = model.half()  # Use FP16 to save memory
    
    # Create dataset
    dataset = AudioDataset(audio_dir)
    
    # Limit samples if specified
    if max_samples and max_samples < len(dataset):
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        dataset.audio_files = [dataset.audio_files[i] for i in indices]
        print(f"Using {max_samples} randomly sampled files")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Change from 1 to 0 to avoid multiprocessing memory
        collate_fn=collate_fn,
        pin_memory=False  # Disable pin_memory to save memory
    )
    
    all_features = []
    
    print(f"Extracting features from layer {layer}...")
    with torch.no_grad():  # Disable gradient computation
        for batch in tqdm(dataloader, desc="Processing batches"):
            try:
                batch = batch.to(device)
                
                # Convert to half precision if model is half
                if device == "cuda":
                    batch = batch.half()
                
                # Extract features
                outputs = model(batch, output_hidden_states=True)
                hidden_states = outputs.hidden_states[layer]  # Shape: (B, T, D)
                
                # Convert to numpy and append
                features = hidden_states.float().cpu().numpy()  # Convert back to float32 for numpy
                for i in range(features.shape[0]):
                    all_features.append(features[i])
                
                # Clear memory after each batch
                del batch, outputs, hidden_states, features
                if device == "cuda" and len(all_features) % 50 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"\nWarning: Failed to process batch: {e}")
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue
    
    # Check if we have any features
    if not all_features:
        raise RuntimeError("No features were extracted! All batches failed.")
    
    # Concatenate all features
    all_features = np.concatenate(all_features, axis=0)
    print(f"Extracted features shape: {all_features.shape}")
    
    return all_features


def train_kmeans(features: np.ndarray, n_clusters: int = 100, batch_size: int = 10000):
    """Train mini-batch k-means on features."""
    
    print(f"Training k-means with {n_clusters} clusters...")
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=100,
        random_state=42,
        verbose=1
    )
    
    kmeans.fit(features)
    
    print(f"✓ K-means training complete")
    print(f"  Inertia: {kmeans.inertia_:.2f}")
    print(f"  Iterations: {kmeans.n_iter_}")
    
    return kmeans


def main():
    parser = argparse.ArgumentParser(
        description="Extract HuBERT features and train k-means centroids"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        required=True,
        help="Directory with audio files (LibriTTS train-clean-100)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./data/hubert_centroids_100.pkl",
        help="Output path for k-means model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/hubert-base-ls960",
        help="HuBERT model name"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=7,
        help="Layer to extract features from (default: 7)"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=100,
        help="Number of k-means clusters (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,  # Reduced batch size
        help="Batch size for feature extraction"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of audio files to use (default: all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        print(f"Error: Audio directory does not exist: {audio_dir}")
        return
    
    # Extract features
    features = extract_hubert_features(
        audio_dir=audio_dir,
        model_name=args.model_name,
        layer=args.layer,
        batch_size=args.batch_size,
        device=args.device,
        max_samples=args.max_samples
    )
    
    # Train k-means
    kmeans = train_kmeans(features, n_clusters=args.n_clusters)
    
    # Save centroids
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump({
            'centroids': kmeans.cluster_centers_,
            'n_clusters': args.n_clusters,
            'model_name': args.model_name,
            'layer': args.layer
        }, f)
    
    print(f"\n✅ Saved k-means centroids to: {output_path}")
    print(f"   Shape: {kmeans.cluster_centers_.shape}")


if __name__ == "__main__":
    main()
