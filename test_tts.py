"""
Test SpeechT5 TTS independently.
"""

import torch
import soundfile as sf
import argparse
from pathlib import Path

from tts.text_to_speech import TextToSpeech


def test_tts(
    text: str,
    output_path: str,
    model_name: str = "microsoft/speecht5_tts",
    vocoder_name: str = "microsoft/speecht5_hifigan",
    speaker: int = 0,
    device: str = "cuda"
):
    """
    Test text-to-speech.
    
    Args:
        text: Text to synthesize
        output_path: Output audio path
        model_name: TTS model name
        vocoder_name: Vocoder model name
        speaker: Speaker ID (0-109 for default embeddings)
        device: Device to use
    """
    print("="*60)
    print("SpeechT5 TTS Test")
    print("="*60)
    
    device = device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Initialize TTS
    print("[1/2] Loading SpeechT5 TTS model...")
    tts = TextToSpeech(
        model_name=model_name,
        vocoder_name=vocoder_name,
        speaker_embeddings_dataset="Matthijs/cmu-arctic-xvectors",
        default_speaker=speaker,
        device=device
    )
    print(f"  Model: {model_name}")
    print(f"  Vocoder: {vocoder_name}")
    print(f"  Speaker: {speaker}")
    
    # Synthesize
    print(f"\n[2/2] Synthesizing speech...")
    print(f"  Text: \"{text}\"")
    
    audio = tts.synthesize(text)
    
    print(f"  Generated: {len(audio)} samples ({len(audio)/16000:.2f}s)")
    
    # Save
    sf.write(output_path, audio, 16000)
    print(f"\n✓ Saved audio to: {output_path}")
    
    print("\n" + "="*60)
    print("TTS Complete!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Test SpeechT5 TTS"
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tts_speecht5_output.wav",
        help="Output audio file"
    )
    parser.add_argument(
        "--speaker",
        type=int,
        default=0,
        help="Speaker ID (0-109)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    test_tts(
        text=args.text,
        output_path=args.output,
        speaker=args.speaker,
        device=args.device
    )


if __name__ == "__main__":
    main()
