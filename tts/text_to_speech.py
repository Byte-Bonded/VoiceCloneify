"""
Text-to-Speech module using SpeechT5.
Generates speech from text with speaker embeddings.
"""

import torch
import numpy as np
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
from typing import Optional


class TextToSpeech:
    """
    TTS using SpeechT5 with HiFi-GAN vocoder.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/speecht5_tts",
        vocoder_name: str = "microsoft/speecht5_hifigan",
        speaker_embeddings_dataset: str = "Matthijs/cmu-arctic-xvectors",
        default_speaker: str = "awb",  # Neutral male voice
        device: str = "cuda"
    ):
        """
        Initialize TTS.
        
        Args:
            model_name: TTS model name
            vocoder_name: Vocoder model name
            speaker_embeddings_dataset: Dataset with speaker embeddings
            default_speaker: Default speaker ID
            device: Device to use
        """
        print(f"Loading TTS model: {model_name}")
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load processor, model, and vocoder
        self.processor = SpeechT5Processor.from_pretrained(model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_name).to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_name).to(self.device)
        
        self.model.eval()
        self.vocoder.eval()
        
        # Load speaker embeddings
        print("Loading speaker embeddings...")
        embeddings_dataset = load_dataset(speaker_embeddings_dataset, split="validation")
        
        self.speaker_embeddings = {}
        for item in embeddings_dataset:
            self.speaker_embeddings[item['speaker']] = torch.tensor(
                item['xvector']
            ).unsqueeze(0).to(self.device)
        
        self.default_speaker = default_speaker
        self.sample_rate = 16000
        
        print(f"✓ TTS model loaded on {self.device}")
        print(f"  Available speakers: {list(self.speaker_embeddings.keys())[:10]}...")
        print(f"  Default speaker: {default_speaker}")
    
    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        speaker: Optional[str] = None,
        speaker_embedding: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Synthesize speech from text.
        
        Args:
            text: Input text
            speaker: Speaker ID (if using predefined speakers)
            speaker_embedding: Custom speaker embedding (B, 512)
            
        Returns:
            audio: Generated audio (samples,)
        """
        if not text.strip():
            return np.array([], dtype=np.float32)
        
        # Get speaker embedding
        if speaker_embedding is None:
            speaker_id = speaker if speaker else self.default_speaker
            if speaker_id not in self.speaker_embeddings:
                print(f"Warning: Speaker {speaker_id} not found, using default")
                speaker_id = self.default_speaker
            speaker_embedding = self.speaker_embeddings[speaker_id]
        
        # Tokenize text
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        
        # Generate spectrogram
        spectrogram = self.model.generate_speech(
            inputs["input_ids"],
            speaker_embedding
        )
        
        # Vocoder (spectrogram to waveform)
        with torch.no_grad():
            audio = self.vocoder(spectrogram)
        
        # Convert to numpy
        audio = audio.cpu().numpy()
        
        return audio
    
    def synthesize_batch(
        self,
        texts: list[str],
        speaker: Optional[str] = None,
        speaker_embedding: Optional[torch.Tensor] = None
    ) -> list[np.ndarray]:
        """
        Synthesize speech from multiple texts.
        
        Args:
            texts: List of input texts
            speaker: Speaker ID
            speaker_embedding: Custom speaker embedding
            
        Returns:
            audios: List of generated audio arrays
        """
        audios = []
        
        for text in texts:
            audio = self.synthesize(
                text=text,
                speaker=speaker,
                speaker_embedding=speaker_embedding
            )
            audios.append(audio)
        
        return audios
    
    def set_default_speaker(self, speaker: str):
        """Set default speaker."""
        if speaker in self.speaker_embeddings:
            self.default_speaker = speaker
            print(f"Default speaker set to: {speaker}")
        else:
            print(f"Speaker {speaker} not found!")
    
    def list_speakers(self) -> list[str]:
        """List available speakers."""
        return list(self.speaker_embeddings.keys())


def synthesize_text(
    text: str,
    model_name: str = "microsoft/speecht5_tts",
    vocoder_name: str = "microsoft/speecht5_hifigan",
    speaker: str = "awb",
    device: str = "cuda",
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Synthesize speech from text (convenience function).
    
    Args:
        text: Input text
        model_name: TTS model name
        vocoder_name: Vocoder name
        speaker: Speaker ID
        device: Device to use
        output_path: Optional path to save audio
        
    Returns:
        audio: Generated audio
    """
    tts = TextToSpeech(
        model_name=model_name,
        vocoder_name=vocoder_name,
        default_speaker=speaker,
        device=device
    )
    
    audio = tts.synthesize(text, speaker=speaker)
    
    # Save if path provided
    if output_path:
        import soundfile as sf
        sf.write(output_path, audio, tts.sample_rate)
        print(f"Saved audio to: {output_path}")
    
    return audio


if __name__ == "__main__":
    # Test TTS
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TTS")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", type=str, default="output_tts.wav", help="Output path")
    parser.add_argument("--speaker", type=str, default="awb", help="Speaker ID")
    parser.add_argument("--model", type=str, default="microsoft/speecht5_tts")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Synthesize
    audio = synthesize_text(
        text=args.text,
        speaker=args.speaker,
        device=args.device,
        output_path=args.output
    )
    
    print(f"\nGenerated audio shape: {audio.shape}")
    print(f"Duration: {len(audio) / 16000:.2f} seconds")
