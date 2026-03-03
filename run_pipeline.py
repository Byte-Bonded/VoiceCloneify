"""
Complete pipeline: ASR → MT → TTS → StreamVC
Streaming end-to-end voice cloning with translation.
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np
import soundfile as sf
import librosa
from typing import Optional
import yaml

from asr.streaming_asr import StreamingASR
from mt.translator import MachineTranslator
from tts.text_to_speech import TextToSpeech
from streamvc.model import StreamVC
from streamvc.audio_utils import MelSpectrogramExtractor
from streamvc.f0_extractor import YinF0Extractor


class VoiceClonePipeline:
    """
    Complete ASR → MT → TTS → StreamVC pipeline.
    """
    
    def __init__(
        self,
        config_path: str = "./configs/pipeline_config.yaml",
        streamvc_checkpoint: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Initialize pipeline.
        
        Args:
            config_path: Path to pipeline config
            streamvc_checkpoint: Path to StreamVC checkpoint
            device: Device to use
        """
        print("="*60)
        print("Initializing VoiceCloneify Pipeline")
        print("="*60)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}\n")
        
        # Initialize ASR
        print("[1/4] Loading ASR (Automatic Speech Recognition)...")
        asr_config = self.config['asr']
        self.asr = StreamingASR(
            model_name=asr_config['model_name'],
            device=asr_config['device'],
            compute_type=asr_config['compute_type'],
            beam_size=asr_config['beam_size'],
            chunk_duration=asr_config['streaming']['chunk_duration']
        )
        
        # Initialize MT
        print("\n[2/4] Loading MT (Machine Translation)...")
        mt_config = self.config['mt']
        self.mt = MachineTranslator(
            model_name=mt_config['model_name'],
            src_lang=mt_config['src_lang'],
            tgt_lang=mt_config['tgt_lang'],
            device=mt_config['device'],
            max_length=mt_config['max_length'],
            num_beams=mt_config['num_beams']
        )
        
        # Initialize TTS
        print("\n[3/4] Loading TTS (Text-to-Speech)...")
        tts_config = self.config['tts']
        self.tts = TextToSpeech(
            model_name=tts_config['model_name'],
            vocoder_name=tts_config['vocoder_name'],
            speaker_embeddings_dataset=tts_config['speaker_embeddings'],
            default_speaker=tts_config['default_speaker'],
            device=tts_config['device']
        )
        
        # Initialize StreamVC
        print("\n[4/4] Loading StreamVC (Voice Conversion)...")
        if streamvc_checkpoint and Path(streamvc_checkpoint).exists():
            streamvc_config_path = self.config['streamvc']['config_path']
            with open(streamvc_config_path, 'r') as f:
                streamvc_config = yaml.safe_load(f)
            
            self.streamvc = StreamVC(streamvc_config).to(self.device)
            checkpoint = torch.load(streamvc_checkpoint, map_location=self.device)
            self.streamvc.load_state_dict(checkpoint['model'])
            self.streamvc.eval()
            
            # Audio processors for StreamVC
            self.mel_extractor = MelSpectrogramExtractor(
                sample_rate=16000,
                hop_length=320,
                n_mels=80
            ).to(self.device)
            
            self.f0_extractor = YinF0Extractor(
                sample_rate=16000,
                frame_length=20.0,
                whitening=True
            )
            
            print("✓ StreamVC loaded from checkpoint")
        else:
            print("⚠ StreamVC checkpoint not found - using TTS output directly")
            self.streamvc = None
        
        self.sample_rate = self.config['pipeline']['sample_rate']
        
        print("\n" + "="*60)
        print("Pipeline Initialized Successfully!")
        print("="*60)
    
    def process_audio(
        self,
        input_audio_path: str,
        target_speaker_audio: Optional[str] = None,
        output_path: str = "output.wav"
    ) -> np.ndarray:
        """
        Process audio through complete pipeline.
        
        Args:
            input_audio_path: Path to input audio (source language)
            target_speaker_audio: Path to target speaker reference (for VC)
            output_path: Path to save output
            
        Returns:
            output_audio: Converted audio
        """
        print("\n" + "="*60)
        print("Processing Pipeline")
        print("="*60)
        
        # Step 1: ASR
        print("\n[Step 1/4] ASR: Transcribing audio...")
        source_audio, sr = librosa.load(input_audio_path, sr=16000, mono=True)
        
        transcript = self.asr.transcribe_chunk(source_audio, language="en")
        print(f"  Transcript (EN): {transcript}")
        
        if not transcript.strip():
            print("⚠ No speech detected!")
            return np.array([])
        
        # Step 2: MT
        print("\n[Step 2/4] MT: Translating text...")
        translation = self.mt.translate_single(transcript)
        print(f"  Translation (FR): {translation}")
        
        # Step 3: TTS
        print("\n[Step 3/4] TTS: Synthesizing speech...")
        # Chunk text to avoid SpeechT5's ~600-token limit
        tts_config = self.config.get('tts', {}).get('synthesis', {})
        chunk_size = tts_config.get('chunk_size', 50)  # words per chunk
        words = translation.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        tts_chunks = [self.tts.synthesize(chunk) for chunk in chunks]
        tts_audio = np.concatenate(tts_chunks) if len(tts_chunks) > 1 else tts_chunks[0]
        print(f"  Generated audio: {len(tts_audio)} samples ({len(tts_audio)/16000:.2f}s) from {len(chunks)} chunk(s)")
        
        # Save TTS output
        tts_output_path = output_path.replace('.wav', '_tts_speecht5.wav')
        sf.write(tts_output_path, tts_audio, self.sample_rate)
        print(f"  Saved TTS output to: {tts_output_path}")
        
        # Step 4: StreamVC (if available)
        if self.streamvc and target_speaker_audio:
            print("\n[Step 4/4] StreamVC: Converting voice...")
            
            # Load target speaker reference
            target_audio, _ = librosa.load(target_speaker_audio, sr=16000, mono=True)
            
            # Prepare inputs
            tts_audio_tensor = torch.from_numpy(tts_audio).unsqueeze(0).unsqueeze(0).to(self.device)
            target_audio_tensor = torch.from_numpy(target_audio).unsqueeze(0).to(self.device)
            
            # Extract target mel (full reference — speaker encoder pools to 1 embedding)
            target_mel = self.mel_extractor(target_audio_tensor)
            
            # Extract f0 features from TTS audio (must stay full length to match content)
            f0_features = self.f0_extractor(tts_audio, return_numpy=False)
            f0_features = f0_features.transpose(0, 1).unsqueeze(0).to(self.device)
            # NOTE: do NOT align target_mel with f0_features — they are from different
            # audio clips. StreamDecoder aligns content↔f0 internally.
            
            # Apply voice conversion
            with torch.no_grad():
                converted_audio, _ = self.streamvc(
                    tts_audio_tensor,
                    target_mel,
                    f0_features
                )
            
            output_audio = converted_audio.squeeze().cpu().numpy()
            print(f"  Converted audio: {len(output_audio)} samples")
        else:
            if not self.streamvc:
                print("\n[Step 4/4] StreamVC: Not available, using TTS output")
            else:
                print("\n[Step 4/4] StreamVC: No target speaker provided, using TTS output")
            output_audio = tts_audio
        
        # Save output
        sf.write(output_path, output_audio, self.sample_rate)
        print(f"\n✓ Saved output to: {output_path}")
        
        return output_audio


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="VoiceCloneify: ASR → MT → TTS → StreamVC Pipeline"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input audio file (English speech)"
    )
    parser.add_argument(
        "--target-speaker",
        type=str,
        default=None,
        help="Target speaker reference audio (for voice conversion)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output audio file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/pipeline_config.yaml",
        help="Pipeline config file"
    )
    parser.add_argument(
        "--streamvc-checkpoint",
        type=str,
        default=None,
        help="StreamVC model checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = VoiceClonePipeline(
        config_path=args.config,
        streamvc_checkpoint=args.streamvc_checkpoint,
        device=args.device
    )
    
    # Process audio
    output_audio = pipeline.process_audio(
        input_audio_path=args.input,
        target_speaker_audio=args.target_speaker,
        output_path=args.output
    )
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
