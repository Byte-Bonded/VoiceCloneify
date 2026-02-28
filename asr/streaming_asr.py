"""
ASR (Automatic Speech Recognition) module using Distil-Whisper.
Supports streaming inference for real-time transcription.
"""

import asyncio
import numpy as np
import torch
try:
    from faster_whisper import WhisperModel
    USING_FASTER_WHISPER = True
except ImportError:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    USING_FASTER_WHISPER = False
    print("Note: faster_whisper not available, using standard Whisper (slower)")
from typing import Optional, AsyncGenerator


class StreamingASR:
    """
    Streaming ASR using faster-whisper (Distil-Whisper).
    """
    
    def __init__(
        self,
        model_name: str = "distil-whisper/distil-small.en",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 5,
        chunk_duration: float = 1.0,  # seconds
        min_silence_duration: float = 0.5
    ):
        """
        Initialize streaming ASR.
        
        Args:
            model_name: Model name or path
            device: Device to run on (cuda/cpu)
            compute_type: Compute type (float16, int8, float32)
            beam_size: Beam search size
            chunk_duration: Duration of audio chunks in seconds
            min_silence_duration: Min silence to trigger sentence boundary
        """
        print(f"Loading ASR model: {model_name}")
        
        # Load model with faster-whisper or standard transformers
        if USING_FASTER_WHISPER:
            self.model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type
            )
            self.processor = None
        else:
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
            if device == "cuda" and compute_type == "float16":
                self.model = self.model.half()
        
        self.device = device
        self.compute_type = compute_type
        
        self.chunk_duration = chunk_duration
        self.min_silence_duration = min_silence_duration
        self.beam_size = beam_size
        self.sample_rate = 16000
        
        # Buffer for accumulating audio
        self.audio_buffer = np.array([], dtype=np.float32)
        self.text_buffer = ""
        
        print(f"✓ ASR model loaded on {device}")
    
    def transcribe_chunk(
        self,
        audio: np.ndarray,
        language: str = "en"
    ) -> str:
        """
        Transcribe a single audio chunk.
        
        Args:
            audio: Audio array (samples,)
            language: Language code
            
        Returns:
            transcript: Transcribed text
        """
        if USING_FASTER_WHISPER:
            # Transcribe with faster-whisper
            segments, info = self.model.transcribe(
                audio,
                language=language,
                beam_size=self.beam_size,
                vad_filter=True,  # Voice activity detection
                vad_parameters=dict(
                    min_silence_duration_ms=int(self.min_silence_duration * 1000)
                )
            )
            
            # Concatenate all segments
            transcript = " ".join([segment.text.strip() for segment in segments])
        else:
            # Use standard Whisper
            input_features = self.processor(
                audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            ).input_features.to(self.device)
            
            if str(self.device).startswith("cuda") and self.compute_type == "float16":
                input_features = input_features.half()
            
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")
            predicted_ids = self.model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            transcript = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcript
    
    def add_audio(self, audio_chunk: np.ndarray) -> Optional[str]:
        """
        Add audio chunk and return transcript if available.
        
        Args:
            audio_chunk: Audio samples to add
            
        Returns:
            transcript: Partial transcript if sentence boundary detected, else None
        """
        # Add to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        
        # Check if we have enough audio
        if len(self.audio_buffer) < self.sample_rate * self.chunk_duration:
            return None
        
        # Transcribe accumulated audio
        transcript = self.transcribe_chunk(self.audio_buffer)
        
        # Check for sentence boundary
        if any(transcript.endswith(char) for char in ['.', '!', '?']):
            # Sentence complete - return and reset
            self.text_buffer += " " + transcript
            result = self.text_buffer.strip()
            self.text_buffer = ""
            self.audio_buffer = np.array([], dtype=np.float32)
            return result
        else:
            # Partial transcript - accumulate
            self.text_buffer += " " + transcript
            self.audio_buffer = np.array([], dtype=np.float32)
            return None
    
    def flush(self) -> str:
        """
        Flush remaining audio and return final transcript.
        
        Returns:
            transcript: Final transcript
        """
        if len(self.audio_buffer) > 0:
            transcript = self.transcribe_chunk(self.audio_buffer)
            self.text_buffer += " " + transcript
        
        result = self.text_buffer.strip()
        
        # Reset
        self.audio_buffer = np.array([], dtype=np.float32)
        self.text_buffer = ""
        
        return result
    
    async def stream_transcribe(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        language: str = "en"
    ) -> AsyncGenerator[str, None]:
        """
        Stream transcription from audio stream.
        
        Args:
            audio_stream: Async generator yielding audio chunks
            language: Language code
            
        Yields:
            transcript: Transcribed text segments
        """
        async for audio_chunk in audio_stream:
            transcript = self.add_audio(audio_chunk)
            
            if transcript:
                yield transcript
        
        # Flush remaining
        final_transcript = self.flush()
        if final_transcript:
            yield final_transcript


# Synchronous wrapper
def transcribe_file(
    audio_path: str,
    model_name: str = "distil-whisper/distil-small.en",
    device: str = "cuda",
    language: str = "en"
) -> str:
    """
    Transcribe an audio file.
    
    Args:
        audio_path: Path to audio file
        model_name: Model name
        device: Device to use
        language: Language code
        
    Returns:
        transcript: Full transcript
    """
    import librosa
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Create ASR
    asr = StreamingASR(model_name=model_name, device=device)
    
    # Transcribe
    transcript = asr.transcribe_chunk(audio, language=language)
    
    return transcript


if __name__ == "__main__":
    # Test ASR
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ASR")
    parser.add_argument("--audio", type=str, required=True, help="Audio file path")
    parser.add_argument("--model", type=str, default="distil-whisper/distil-small.en")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Transcribe
    transcript = transcribe_file(
        audio_path=args.audio,
        model_name=args.model,
        device=args.device
    )
    
    print(f"\nTranscript: {transcript}")
