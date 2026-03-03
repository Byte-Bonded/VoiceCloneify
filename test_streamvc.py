"""
Test StreamVC voice conversion independently.
"""

import torch
import yaml
import librosa
import soundfile as sf
import argparse
from pathlib import Path

from streamvc.model import StreamVC
from streamvc.audio_utils import MelSpectrogramExtractor
from streamvc.f0_extractor import F0ExtractorModule


def load_audio(path, sr=16000):
    """Load audio file."""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return torch.from_numpy(audio).float()


def test_voice_conversion(
    source_path: str,
    target_path: str,
    output_path: str,
    checkpoint_path: str,
    config_path: str = "./configs/streamvc_config.yaml",
    device: str = "cuda"
):
    """
    Test voice conversion.
    
    Args:
        source_path: Source audio (content)
        target_path: Target speaker reference
        output_path: Output audio path
        checkpoint_path: StreamVC checkpoint
        config_path: StreamVC config
        device: Device to use
    """
    print("="*60)
    print("StreamVC Voice Conversion Test")
    print("="*60)
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Load config
    print("[1/5] Loading config...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    print("[2/5] Loading StreamVC model...")
    model = StreamVC(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"  Loaded checkpoint: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Step: {checkpoint.get('step', 'unknown')}")
    
    # Initialize audio processors
    print("\n[3/5] Initializing audio processors...")
    mel_extractor = MelSpectrogramExtractor(
        sample_rate=config['audio']['sample_rate'],
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length'],
        n_mels=config['audio']['n_mels']
    ).to(device)
    
    f0_extractor = F0ExtractorModule(
        sample_rate=config['audio']['sample_rate'],
        frame_length=config['model']['f0_extractor']['frame_length'],
        f0_min=config['model']['f0_extractor']['f0_min'],
        f0_max=config['model']['f0_extractor']['f0_max'],
        whitening=config['model']['f0_extractor']['whitening']
    )
    
    # Load audio
    print("\n[4/5] Loading audio files...")
    source_audio = load_audio(source_path, config['audio']['sample_rate'])
    target_audio = load_audio(target_path, config['audio']['sample_rate'])
    print(f"  Source: {source_path} ({len(source_audio)} samples)")
    print(f"  Target: {target_path} ({len(target_audio)} samples)")
    
    # Prepare inputs
    source_audio = source_audio.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, T)
    target_audio = target_audio.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, T)
    
    # Extract features
    print("\n[5/5] Converting voice...")
    with torch.no_grad():
        # Extract target mel
        target_mel = mel_extractor(target_audio)
        
        # Extract F0 from source
        f0_features = f0_extractor(source_audio)
        
        # Align temporal dimensions
        min_len = min(target_mel.size(2), f0_features.size(2))
        target_mel = target_mel[:, :, :min_len]
        f0_features = f0_features[:, :, :min_len]
        
        # Forward pass
        converted_audio, _ = model(source_audio, target_mel, f0_features)
        
        # Convert to numpy
        converted_audio = converted_audio.squeeze().cpu().numpy()
    
    # Save output
    sf.write(output_path, converted_audio, config['audio']['sample_rate'])
    print(f"\n✓ Saved converted audio to: {output_path}")
    print(f"  Duration: {len(converted_audio) / config['audio']['sample_rate']:.2f}s")
    
    print("\n" + "="*60)
    print("Voice Conversion Complete!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Test StreamVC voice conversion"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source audio (content to convert)"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target speaker reference audio"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="converted_output.wav",
        help="Output audio file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="StreamVC checkpoint path"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/streamvc_config.yaml",
        help="StreamVC config file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    test_voice_conversion(
        source_path=args.source,
        target_path=args.target,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )


if __name__ == "__main__":
    main()
