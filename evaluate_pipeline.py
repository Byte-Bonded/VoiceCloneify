#!/usr/bin/env python3
"""
VoiceCloneify — Comprehensive Pipeline Evaluation

Evaluates the full ASR → MT → TTS → StreamVC pipeline with per-stage and
end-to-end metrics for speech recognition, translation, synthesis quality,
and voice cloning fidelity.

Metrics computed
================
  ASR   : WER (Word Error Rate), CER (Character Error Rate)
  MT    : BLEU score
  TTS   : PESQ (Perceptual Evaluation of Speech Quality)
  Voice : Speaker Cosine Similarity (resemblyzer d-vector)
          Speaker Cosine Similarity (StreamVC internal encoder)
          F0 Pearson Correlation
          F0 RMSE (Hz)
          Voicing Decision Accuracy
          Mel Cepstral Distortion (MCD, dB)
          Energy Correlation
          Duration Ratio
  E2E   : PESQ of final output, SNR

Usage
=====
  # Auto-generate test set from LibriTTS
  python evaluate_pipeline.py --auto-testset --num-samples 50 --device cuda

  # Custom test set
  python evaluate_pipeline.py --test-dir data/test_sets/my_test \
      --device cuda --streamvc-checkpoint checkpoints/streamvc/best.pt

  # Quick test (no StreamVC)
  python evaluate_pipeline.py --auto-testset --num-samples 5 --device cpu
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml

# ── Suppress non-critical warnings ──────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.utils.weight_norm")

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# Metric helpers — imported lazily to give clear errors if a package is missing
# =============================================================================

def _import_jiwer():
    try:
        import jiwer
        return jiwer
    except ImportError:
        raise ImportError("jiwer is required for WER/CER.  pip install jiwer")


def _import_sacrebleu():
    try:
        import sacrebleu
        return sacrebleu
    except ImportError:
        raise ImportError("sacrebleu is required for BLEU.  pip install sacrebleu")


def _import_pesq():
    try:
        from pesq import pesq as pesq_fn
        return pesq_fn
    except ImportError:
        raise ImportError("pesq is required.  pip install pesq")


def _import_resemblyzer():
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        return VoiceEncoder, preprocess_wav
    except ImportError:
        raise ImportError("resemblyzer is required.  pip install resemblyzer")


# =============================================================================
# Individual metric functions
# =============================================================================

# ── ASR metrics ──────────────────────────────────────────────────────────────

def compute_wer(hypothesis: str, reference: str) -> float:
    """Compute Word Error Rate."""
    jiwer = _import_jiwer()
    return jiwer.wer(reference, hypothesis)


def compute_cer(hypothesis: str, reference: str) -> float:
    """Compute Character Error Rate."""
    jiwer = _import_jiwer()
    return jiwer.cer(reference, hypothesis)


# ── MT metrics ───────────────────────────────────────────────────────────────

def compute_bleu(hypothesis: str, reference: str) -> float:
    """Compute sentence-level BLEU score."""
    sacrebleu = _import_sacrebleu()
    result = sacrebleu.sentence_bleu(hypothesis, [reference])
    return result.score


# ── Speech quality metrics ───────────────────────────────────────────────────

def compute_pesq_score(reference: np.ndarray, degraded: np.ndarray,
                       sample_rate: int = 16000) -> float:
    """
    Compute PESQ (narrowband mode for 16 kHz).
    Returns score in range [-0.5, 4.5].
    Returns NaN if signals are too short or incompatible.
    """
    pesq_fn = _import_pesq()
    try:
        # Align lengths
        min_len = min(len(reference), len(degraded))
        if min_len < sample_rate:  # need at least 1 second
            return float('nan')
        ref = reference[:min_len].astype(np.float64)
        deg = degraded[:min_len].astype(np.float64)
        return pesq_fn(sample_rate, ref, deg, 'nb')
    except Exception:
        return float('nan')


def compute_snr(signal: np.ndarray, noise_floor_db: float = -60.0) -> float:
    """Estimate SNR of a signal (dB) using energy."""
    sig_power = np.mean(signal ** 2)
    if sig_power < 1e-10:
        return float('nan')
    return 10.0 * np.log10(sig_power / (10 ** (noise_floor_db / 10)))


# ── Speaker similarity ──────────────────────────────────────────────────────

class SpeakerSimilarityEvaluator:
    """Compute speaker cosine similarity using resemblyzer d-vectors."""

    def __init__(self):
        VoiceEncoder, _ = _import_resemblyzer()
        self.encoder = VoiceEncoder()
        _, self.preprocess = _import_resemblyzer()

    def embed(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Get d-vector embedding (256-dim)."""
        wav = self.preprocess(audio, source_sr=sr)
        return self.encoder.embed_utterance(wav)

    def similarity(self, audio_a: np.ndarray, audio_b: np.ndarray,
                   sr: int = 16000) -> float:
        """Cosine similarity between two utterances."""
        emb_a = self.embed(audio_a, sr)
        emb_b = self.embed(audio_b, sr)
        cos = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8)
        return float(cos)


class StreamVCSpeakerSimilarity:
    """Compute speaker similarity using StreamVC's own SpeakerEncoder."""

    def __init__(self, model, mel_extractor, device):
        self.model = model
        self.mel_extractor = mel_extractor
        self.device = device

    @torch.no_grad()
    def embed(self, audio: np.ndarray) -> np.ndarray:
        """Get StreamVC speaker embedding (256-dim)."""
        t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(self.device)
        mel = self.mel_extractor(t)
        emb = self.model.speaker_encoder(mel)
        return emb.squeeze(0).cpu().numpy()

    def similarity(self, audio_a: np.ndarray, audio_b: np.ndarray) -> float:
        emb_a = self.embed(audio_a)
        emb_b = self.embed(audio_b)
        cos = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8)
        return float(cos)


# ── Prosodic / spectral metrics ─────────────────────────────────────────────

def extract_f0(audio: np.ndarray, sr: int = 16000,
               fmin: float = 50.0, fmax: float = 600.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract F0 contour using parselmouth (Praat).
    Returns (f0, voiced_mask) where unvoiced frames have f0 = 0.
    """
    import parselmouth
    snd = parselmouth.Sound(audio, sampling_frequency=sr)
    pitch = snd.to_pitch_ac(
        time_step=0.01,
        pitch_floor=fmin,
        pitch_ceiling=fmax
    )
    f0 = pitch.selected_array['frequency']
    voiced = f0 > 0
    return f0, voiced


def compute_f0_metrics(f0_output: np.ndarray, voiced_output: np.ndarray,
                       f0_target: np.ndarray, voiced_target: np.ndarray) -> Dict[str, float]:
    """
    Compute F0-related metrics between output and target.
    """
    min_len = min(len(f0_output), len(f0_target))
    f0_o = f0_output[:min_len]
    f0_t = f0_target[:min_len]
    v_o = voiced_output[:min_len]
    v_t = voiced_target[:min_len]

    # Both-voiced mask
    both_voiced = v_o & v_t

    metrics = {}

    # F0 Pearson correlation (on voiced frames)
    if both_voiced.sum() > 2:
        corr = np.corrcoef(f0_o[both_voiced], f0_t[both_voiced])[0, 1]
        metrics['f0_pearson_corr'] = float(corr) if not np.isnan(corr) else 0.0
    else:
        metrics['f0_pearson_corr'] = 0.0

    # F0 RMSE in Hz (on voiced frames)
    if both_voiced.sum() > 0:
        rmse = np.sqrt(np.mean((f0_o[both_voiced] - f0_t[both_voiced]) ** 2))
        metrics['f0_rmse_hz'] = float(rmse)
    else:
        metrics['f0_rmse_hz'] = float('nan')

    # F0 RMSE in cents (log-scale, more perceptually meaningful)
    if both_voiced.sum() > 0:
        cents_o = 1200.0 * np.log2(f0_o[both_voiced] + 1e-8)
        cents_t = 1200.0 * np.log2(f0_t[both_voiced] + 1e-8)
        metrics['f0_rmse_cents'] = float(np.sqrt(np.mean((cents_o - cents_t) ** 2)))
    else:
        metrics['f0_rmse_cents'] = float('nan')

    # Voicing decision accuracy
    metrics['voicing_accuracy'] = float(np.mean(v_o[:min_len] == v_t[:min_len]))

    # Voiced frame ratio (should be similar)
    metrics['voiced_ratio_output'] = float(v_o.mean())
    metrics['voiced_ratio_target'] = float(v_t.mean())

    return metrics


def compute_mcd(output_audio: np.ndarray, target_audio: np.ndarray,
                sr: int = 16000, n_mfcc: int = 13) -> float:
    """
    Compute Mel Cepstral Distortion (MCD) in dB.
    
    Lower is better. Typical values: 4-8 dB (good), >10 dB (poor).
    """
    # Extract MFCCs
    mfcc_out = librosa.feature.mfcc(y=output_audio, sr=sr, n_mfcc=n_mfcc,
                                     hop_length=256, n_fft=1024)
    mfcc_tgt = librosa.feature.mfcc(y=target_audio, sr=sr, n_mfcc=n_mfcc,
                                     hop_length=256, n_fft=1024)

    # Align lengths
    min_frames = min(mfcc_out.shape[1], mfcc_tgt.shape[1])
    mfcc_out = mfcc_out[:, :min_frames]
    mfcc_tgt = mfcc_tgt[:, :min_frames]

    # MCD (exclude C0)
    diff = mfcc_out[1:, :] - mfcc_tgt[1:, :]
    mcd = (10.0 / np.log(10.0)) * np.sqrt(2.0) * np.mean(np.sqrt(np.sum(diff ** 2, axis=0)))
    return float(mcd)


def compute_energy_correlation(output_audio: np.ndarray,
                                target_audio: np.ndarray,
                                sr: int = 16000,
                                hop_length: int = 256) -> float:
    """Pearson correlation of frame-level RMS energy."""
    rms_out = librosa.feature.rms(y=output_audio, hop_length=hop_length).flatten()
    rms_tgt = librosa.feature.rms(y=target_audio, hop_length=hop_length).flatten()

    min_len = min(len(rms_out), len(rms_tgt))
    if min_len < 2:
        return 0.0
    corr = np.corrcoef(rms_out[:min_len], rms_tgt[:min_len])[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def compute_duration_ratio(output_audio: np.ndarray,
                           target_audio: np.ndarray) -> float:
    """Duration ratio: output_duration / target_duration."""
    if len(target_audio) == 0:
        return float('nan')
    return len(output_audio) / len(target_audio)


# =============================================================================
# Test set generation / loading
# =============================================================================

def generate_libritts_testset(data_dir: str, num_samples: int = 50,
                               seed: int = 42) -> List[Dict]:
    """
    Generate a test set from LibriTTS.
    
    Each sample has: audio_path, transcript (English), speaker_id.
    Reads transcripts from the .normalized.txt files that ship with LibriTTS.
    """
    rng = np.random.RandomState(seed)
    data_path = Path(data_dir)

    # Find all transcript files
    trans_files = sorted(data_path.rglob("*.normalized.txt"))
    if not trans_files:
        # Fallback: try .original.txt
        trans_files = sorted(data_path.rglob("*.original.txt"))
    if not trans_files:
        raise FileNotFoundError(
            f"No LibriTTS transcript files found in {data_dir}. "
            "Download LibriTTS first: python data/download_datasets.py"
        )

    # Build (audio_path, transcript, speaker_id) tuples
    samples = []
    for tf in trans_files:
        transcript = tf.read_text().strip()
        if not transcript:
            continue
        # Derive audio path: strip the double suffix (.normalized.txt → .wav)
        # tf.stem is e.g. "103_1241_000000_000001.normalized", so remove that extra suffix
        base_stem = tf.stem  # e.g. "103_1241_000000_000001.normalized"
        if base_stem.endswith('.normalized') or base_stem.endswith('.original'):
            base_stem = base_stem[:base_stem.rfind('.')]
        wav_path = tf.parent / (base_stem + '.wav')
        if not wav_path.exists():
            wav_path = tf.parent / (base_stem + '.flac')
        if not wav_path.exists():
            continue
        # Speaker ID from LibriTTS path structure: .../<speaker_id>/<chapter_id>/...
        parts = wav_path.relative_to(data_path).parts
        speaker_id = parts[0] if len(parts) >= 2 else "unknown"
        samples.append({
            'audio_path': str(wav_path),
            'transcript_en': transcript,
            'speaker_id': speaker_id
        })

    if len(samples) == 0:
        raise FileNotFoundError(f"No valid audio+transcript pairs in {data_dir}")

    # Subsample
    indices = rng.choice(len(samples), size=min(num_samples, len(samples)), replace=False)
    selected = [samples[i] for i in indices]

    print(f"  Generated test set: {len(selected)} samples from {data_dir}")
    print(f"  Unique speakers: {len(set(s['speaker_id'] for s in selected))}")
    return selected


def load_custom_testset(test_dir: str) -> List[Dict]:
    """
    Load a custom test set.
    
    Expected structure:
        test_dir/
            audio/          # .wav files
            references.json # {"filename": {"transcript_en": ..., "transcript_fr": ..., "speaker_id": ...}}
    
    If references.json doesn't exist, only voice-cloning metrics (no WER/BLEU) are computed.
    """
    test_path = Path(test_dir)

    # Find audio files
    audio_dir = test_path / "audio"
    if not audio_dir.exists():
        audio_dir = test_path  # Fallback: files directly in test_dir

    audio_files = sorted(audio_dir.glob("*.wav")) + sorted(audio_dir.glob("*.flac"))
    if not audio_files:
        raise FileNotFoundError(f"No audio files in {audio_dir}")

    # Load references if available
    ref_path = test_path / "references.json"
    references = {}
    if ref_path.exists():
        with open(ref_path) as f:
            references = json.load(f)

    samples = []
    for af in audio_files:
        entry = {'audio_path': str(af)}
        fname = af.stem
        if fname in references:
            entry.update(references[fname])
        samples.append(entry)

    print(f"  Loaded custom test set: {len(samples)} samples")
    print(f"  With references: {len(references)} / {len(samples)}")
    return samples


# =============================================================================
# Main evaluator
# =============================================================================

class PipelineEvaluator:
    """Evaluates the full VoiceCloneify pipeline with comprehensive metrics."""

    def __init__(
        self,
        config_path: str = "configs/pipeline_config.yaml",
        streamvc_checkpoint: Optional[str] = None,
        streamvc_config_path: str = "configs/streamvc_config.yaml",
        target_speaker_audio: Optional[str] = None,
        device: str = "cuda",
        output_dir: str = "outputs/eval",
        skip_mt: bool = False
    ):
        self.skip_mt = skip_mt
        self.device_str = device if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sr = 16000
        self.target_speaker_audio_path = target_speaker_audio

        # ---- Load pipeline config ----
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # ---- Initialize pipeline components ----
        print("=" * 60)
        print("  VoiceCloneify Pipeline Evaluator")
        print("=" * 60)

        # ASR
        print("[1/5] Loading ASR...")
        from asr.streaming_asr import StreamingASR
        asr_cfg = self.config['asr']
        self.asr = StreamingASR(
            model_name=asr_cfg['model_name'],
            device=self.device_str,
            compute_type=asr_cfg.get('compute_type', 'float16')
        )

        # MT
        if self.skip_mt:
            print("[2/5] Loading MT... SKIPPED (--skip-mt flag set)")
            self.mt = None
        else:
            print("[2/5] Loading MT...")
            from mt.translator import MachineTranslator
            mt_cfg = self.config['mt']
            self.mt = MachineTranslator(
                model_name=mt_cfg['model_name'],
                src_lang=mt_cfg['src_lang'],
                tgt_lang=mt_cfg['tgt_lang'],
                device=self.device_str,
                max_length=mt_cfg.get('max_length', 256),
                num_beams=mt_cfg.get('num_beams', 5)
            )

        # TTS
        print("[3/5] Loading TTS...")
        from tts.text_to_speech import TextToSpeech
        tts_cfg = self.config['tts']
        self.tts = TextToSpeech(
            model_name=tts_cfg['model_name'],
            vocoder_name=tts_cfg['vocoder_name'],
            speaker_embeddings_dataset=tts_cfg.get(
                'speaker_embeddings', 'Matthijs/cmu-arctic-xvectors'),
            default_speaker=tts_cfg.get('default_speaker', 'awb'),
            device=self.device_str
        )

        # StreamVC (optional)
        self.streamvc = None
        self.streamvc_mel = None
        self.streamvc_f0 = None
        self.streamvc_spk_sim = None

        if streamvc_checkpoint and Path(streamvc_checkpoint).exists():
            print("[4/5] Loading StreamVC...")
            from streamvc.model import StreamVC
            from streamvc.audio_utils import MelSpectrogramExtractor
            from streamvc.f0_extractor import YinF0Extractor

            with open(streamvc_config_path) as f:
                svc_config = yaml.safe_load(f)

            self.streamvc = StreamVC(svc_config).to(self.device)
            ckpt = torch.load(streamvc_checkpoint, map_location=self.device, weights_only=False)
            self.streamvc.load_state_dict(ckpt['model'])
            self.streamvc.eval()

            self.streamvc_mel = MelSpectrogramExtractor(
                sample_rate=svc_config['audio']['sample_rate'],
                n_fft=svc_config['audio']['n_fft'],
                hop_length=svc_config['audio']['hop_length'],
                n_mels=svc_config['audio']['n_mels']
            ).to(self.device)

            self.streamvc_f0 = YinF0Extractor(
                sample_rate=svc_config['audio']['sample_rate'],
                frame_length=svc_config['model']['f0_extractor']['frame_length'],
                f0_min=svc_config['model']['f0_extractor']['f0_min'],
                f0_max=svc_config['model']['f0_extractor']['f0_max'],
                whitening=svc_config['model']['f0_extractor']['whitening']
            )

            self.streamvc_spk_sim = StreamVCSpeakerSimilarity(
                self.streamvc, self.streamvc_mel, self.device
            )

            # Reusable F0 extractor module for voice conversion
            from streamvc.f0_extractor import F0ExtractorModule
            self._f0_module = F0ExtractorModule(
                sample_rate=svc_config['audio']['sample_rate'],
                frame_length=svc_config['model']['f0_extractor']['frame_length'],
                f0_min=svc_config['model']['f0_extractor']['f0_min'],
                f0_max=svc_config['model']['f0_extractor']['f0_max'],
                whitening=svc_config['model']['f0_extractor']['whitening']
            )

            print(f"  StreamVC loaded from {streamvc_checkpoint}")
        else:
            print("[4/5] StreamVC — skipped (no checkpoint)")

        # External speaker similarity (resemblyzer)
        print("[5/5] Loading speaker encoder (resemblyzer)...")
        try:
            self.spk_evaluator = SpeakerSimilarityEvaluator()
        except ImportError:
            print("  WARNING: resemblyzer not installed. Speaker similarity disabled.")
            self.spk_evaluator = None

        # Load target speaker reference audio
        self.target_speaker_audio = None
        if target_speaker_audio and Path(target_speaker_audio).exists():
            self.target_speaker_audio, _ = librosa.load(
                target_speaker_audio, sr=self.sr, mono=True)
            print(f"  Target speaker: {target_speaker_audio} "
                  f"({len(self.target_speaker_audio)/self.sr:.1f}s)")

        print()

    # ── Helper: run StreamVC voice conversion ────────────────────────────────

    @torch.no_grad()
    def _voice_convert(self, tts_audio: np.ndarray,
                       target_audio: np.ndarray) -> np.ndarray:
        """Run StreamVC voice conversion."""
        if self.streamvc is None:
            return tts_audio

        source = torch.from_numpy(tts_audio).float().unsqueeze(0).unsqueeze(0).to(self.device)
        target = torch.from_numpy(target_audio).float().unsqueeze(0).unsqueeze(0).to(self.device)

        target_mel = self.streamvc_mel(target)

        f0_features = self._f0_module(source)

        min_len = min(target_mel.size(2), f0_features.size(2))
        target_mel = target_mel[:, :, :min_len]
        f0_features = f0_features[:, :, :min_len]

        output, _ = self.streamvc(source, target_mel, f0_features)
        return output.squeeze().cpu().numpy()

    # ── Per-sample evaluation ────────────────────────────────────────────────

    def evaluate_sample(self, sample: Dict, idx: int) -> Dict:
        """Evaluate a single sample through the full pipeline."""
        result = {'index': idx, 'audio_path': sample.get('audio_path', '')}

        # Load input audio
        audio, _ = librosa.load(sample['audio_path'], sr=self.sr, mono=True)
        result['input_duration_s'] = len(audio) / self.sr

        # ── ASR ──────────────────────────────────────────────────────────────
        t0 = time.time()
        transcript = self.asr.transcribe_chunk(audio, language="en")
        result['asr_time_s'] = time.time() - t0
        result['asr_output'] = transcript

        # Save ASR output
        asr_dir = self.output_dir / "asr"
        asr_dir.mkdir(exist_ok=True)
        (asr_dir / f"sample_{idx:04d}.txt").write_text(transcript)

        # ASR metrics (if ground truth available)
        if 'transcript_en' in sample and sample['transcript_en']:
            ref = sample['transcript_en']
            result['wer'] = compute_wer(transcript, ref)
            result['cer'] = compute_cer(transcript, ref)
            result['asr_reference'] = ref

        # ── MT ───────────────────────────────────────────────────────────────
        if self.mt is not None:
            t0 = time.time()
            translation = self.mt.translate_single(transcript)
            result['mt_time_s'] = time.time() - t0
            result['mt_output'] = translation
        else:
            # MT skipped — use ASR transcript directly for TTS (English)
            translation = transcript
            result['mt_skipped'] = True
            result['mt_output'] = translation

        mt_dir = self.output_dir / "mt"
        mt_dir.mkdir(exist_ok=True)
        (mt_dir / f"sample_{idx:04d}.txt").write_text(translation)

        # MT metrics (if ground truth available)
        if 'transcript_fr' in sample and sample['transcript_fr']:
            result['bleu'] = compute_bleu(translation, sample['transcript_fr'])
            result['mt_reference'] = sample['transcript_fr']

        # ── TTS ──────────────────────────────────────────────────────────────
        t0 = time.time()
        tts_audio = self.tts.synthesize(translation)
        result['tts_time_s'] = time.time() - t0
        result['tts_duration_s'] = len(tts_audio) / self.sr

        tts_dir = self.output_dir / "tts"
        tts_dir.mkdir(exist_ok=True)
        sf.write(str(tts_dir / f"sample_{idx:04d}.wav"), tts_audio, self.sr)

        # TTS audio quality (SNR)
        result['tts_snr_db'] = compute_snr(tts_audio)

        # ── StreamVC (optional) ──────────────────────────────────────────────
        final_audio = tts_audio  # default: no voice conversion

        target_ref = self.target_speaker_audio
        if target_ref is None:
            # Use the input audio itself as a proxy target speaker
            target_ref = audio

        if self.streamvc is not None:
            t0 = time.time()
            vc_audio = self._voice_convert(tts_audio, target_ref)
            result['vc_time_s'] = time.time() - t0
            result['vc_duration_s'] = len(vc_audio) / self.sr
            final_audio = vc_audio

            vc_dir = self.output_dir / "streamvc"
            vc_dir.mkdir(exist_ok=True)
            sf.write(str(vc_dir / f"sample_{idx:04d}.wav"), vc_audio, self.sr)

            # Speaker similarity — resemblyzer
            if self.spk_evaluator is not None:
                result['speaker_sim_resemblyzer'] = self.spk_evaluator.similarity(
                    vc_audio, target_ref, self.sr)

            # Speaker similarity — StreamVC encoder
            if self.streamvc_spk_sim is not None:
                result['speaker_sim_streamvc'] = self.streamvc_spk_sim.similarity(
                    vc_audio, target_ref)

            # F0 metrics (output vs target speaker reference)
            try:
                f0_out, v_out = extract_f0(vc_audio, self.sr)
                f0_tgt, v_tgt = extract_f0(target_ref, self.sr)
                f0_metrics = compute_f0_metrics(f0_out, v_out, f0_tgt, v_tgt)
                result.update(f0_metrics)
            except Exception as e:
                result['f0_error'] = str(e)

            # MCD (output vs target speaker)
            try:
                result['mcd_db'] = compute_mcd(vc_audio, target_ref, self.sr)
            except Exception as e:
                result['mcd_error'] = str(e)

            # Energy correlation
            try:
                result['energy_correlation'] = compute_energy_correlation(
                    vc_audio, target_ref, self.sr)
            except Exception:
                result['energy_correlation'] = float('nan')

            # Duration ratio
            result['duration_ratio'] = compute_duration_ratio(vc_audio, target_ref)

        # ── End-to-end metrics ───────────────────────────────────────────────
        # Save final output
        final_dir = self.output_dir / "final"
        final_dir.mkdir(exist_ok=True)
        sf.write(str(final_dir / f"sample_{idx:04d}.wav"), final_audio, self.sr)

        result['final_snr_db'] = compute_snr(final_audio)
        result['total_time_s'] = (
            result.get('asr_time_s', 0) + result.get('mt_time_s', 0) +
            result.get('tts_time_s', 0) + result.get('vc_time_s', 0)
        )

        # PESQ (final output vs input — cross-lingual, so this is a rough proxy)
        try:
            result['final_pesq'] = compute_pesq_score(audio, final_audio, self.sr)
        except Exception:
            result['final_pesq'] = float('nan')

        return result

    # ── Full evaluation ──────────────────────────────────────────────────────

    def evaluate(self, test_samples: List[Dict]) -> Dict:
        """Run evaluation on the full test set."""
        print("=" * 60)
        print(f"  Evaluating {len(test_samples)} samples")
        print("=" * 60)

        per_sample_results = []

        for idx, sample in enumerate(test_samples):
            print(f"\n[{idx+1}/{len(test_samples)}] {Path(sample['audio_path']).name}")
            try:
                result = self.evaluate_sample(sample, idx)
                per_sample_results.append(result)

                # Print key metrics inline
                parts = []
                if 'wer' in result:
                    parts.append(f"WER={result['wer']:.2%}")
                if 'cer' in result:
                    parts.append(f"CER={result['cer']:.2%}")
                if 'bleu' in result:
                    parts.append(f"BLEU={result['bleu']:.1f}")
                if 'speaker_sim_resemblyzer' in result:
                    parts.append(f"SpkSim={result['speaker_sim_resemblyzer']:.3f}")
                if 'mcd_db' in result:
                    parts.append(f"MCD={result['mcd_db']:.1f}dB")
                if parts:
                    print(f"  → {' | '.join(parts)}")

            except Exception as e:
                print(f"  ERROR: {e}")
                per_sample_results.append({
                    'index': idx,
                    'audio_path': sample.get('audio_path', ''),
                    'error': str(e)
                })

        # ── Aggregate metrics ────────────────────────────────────────────────
        report = self._aggregate(per_sample_results)
        report['per_sample'] = per_sample_results

        # Save full report
        report_path = self.output_dir / "reports" / "metrics_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Print summary
        self._print_summary(report)

        return report

    # ── Aggregation helpers ──────────────────────────────────────────────────

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        clean = [v for v in values if v is not None and not np.isnan(v)]
        return float(np.mean(clean)) if clean else float('nan')

    @staticmethod
    def _safe_std(values: List[float]) -> float:
        clean = [v for v in values if v is not None and not np.isnan(v)]
        return float(np.std(clean)) if len(clean) > 1 else 0.0

    def _aggregate(self, results: List[Dict]) -> Dict:
        """Compute aggregate statistics from per-sample results."""
        # Collect all numeric metrics
        metric_keys = set()
        for r in results:
            for k, v in r.items():
                if isinstance(v, (int, float)) and k not in ('index',):
                    metric_keys.add(k)

        agg = {'num_samples': len(results), 'num_errors': 0}

        for key in sorted(metric_keys):
            values = [r.get(key) for r in results if key in r and 'error' not in r]
            values = [v for v in values if v is not None]
            if values:
                agg[f'{key}_mean'] = self._safe_mean(values)
                agg[f'{key}_std'] = self._safe_std(values)
                agg[f'{key}_min'] = float(np.nanmin(values))
                agg[f'{key}_max'] = float(np.nanmax(values))

        agg['num_errors'] = sum(1 for r in results if 'error' in r)

        return agg

    def _print_summary(self, report: Dict):
        """Print a formatted summary table."""
        print("\n")
        print("=" * 70)
        print("  EVALUATION REPORT")
        print("=" * 70)
        print(f"  Samples evaluated: {report.get('num_samples', 0)}")
        print(f"  Errors           : {report.get('num_errors', 0)}")
        print()

        # Group by category
        categories = [
            ("ASR Metrics", [
                ("WER  (↓ better)", 'wer_mean', 'wer_std', '{:.2%}'),
                ("CER  (↓ better)", 'cer_mean', 'cer_std', '{:.2%}'),
            ]),
            ("MT Metrics", [
                ("BLEU (↑ better)", 'bleu_mean', 'bleu_std', '{:.1f}'),
            ]),
            ("TTS Quality", [
                ("TTS SNR (dB)", 'tts_snr_db_mean', 'tts_snr_db_std', '{:.1f}'),
            ]),
            ("Voice Cloning — Speaker Similarity (↑ better)", [
                ("Resemblyzer Cosine Sim", 'speaker_sim_resemblyzer_mean',
                 'speaker_sim_resemblyzer_std', '{:.3f}'),
                ("StreamVC Encoder Sim", 'speaker_sim_streamvc_mean',
                 'speaker_sim_streamvc_std', '{:.3f}'),
            ]),
            ("Voice Cloning — Prosodic Features", [
                ("F0 Pearson Corr (↑)", 'f0_pearson_corr_mean',
                 'f0_pearson_corr_std', '{:.3f}'),
                ("F0 RMSE (Hz) (↓)", 'f0_rmse_hz_mean',
                 'f0_rmse_hz_std', '{:.1f}'),
                ("F0 RMSE (cents) (↓)", 'f0_rmse_cents_mean',
                 'f0_rmse_cents_std', '{:.1f}'),
                ("Voicing Accuracy (↑)", 'voicing_accuracy_mean',
                 'voicing_accuracy_std', '{:.2%}'),
            ]),
            ("Voice Cloning — Spectral & Energy", [
                ("MCD (dB) (↓ better)", 'mcd_db_mean', 'mcd_db_std', '{:.2f}'),
                ("Energy Correlation (↑)", 'energy_correlation_mean',
                 'energy_correlation_std', '{:.3f}'),
                ("Duration Ratio (→1.0)", 'duration_ratio_mean',
                 'duration_ratio_std', '{:.2f}'),
            ]),
            ("End-to-End", [
                ("Final SNR (dB)", 'final_snr_db_mean', 'final_snr_db_std', '{:.1f}'),
                ("Final PESQ", 'final_pesq_mean', 'final_pesq_std', '{:.2f}'),
                ("Total Time (s)", 'total_time_s_mean', 'total_time_s_std', '{:.2f}'),
            ]),
        ]

        for cat_name, metrics in categories:
            has_any = any(m_key in report for _, m_key, _, _ in metrics)
            if not has_any:
                continue
            print(f"  ┌── {cat_name}")
            for label, m_key, s_key, fmt in metrics:
                if m_key in report:
                    val = report[m_key]
                    std = report.get(s_key, 0)
                    val_str = fmt.format(val) if not np.isnan(val) else "N/A"
                    std_str = fmt.format(std) if not np.isnan(std) else ""
                    line = f"  │  {label:<35s} {val_str:>10s}"
                    if std_str:
                        line += f" ± {std_str}"
                    print(line)
            print("  │")

        report_path = self.output_dir / "reports" / "metrics_report.json"
        print(f"  Full report saved to: {report_path}")
        print("=" * 70)


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VoiceCloneify — Comprehensive Pipeline Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick eval with auto-generated LibriTTS test set (no StreamVC)
  python evaluate_pipeline.py --auto-testset --num-samples 10 --device cpu

  # Full eval with StreamVC
  python evaluate_pipeline.py --auto-testset \\
      --streamvc-checkpoint checkpoints/streamvc/best.pt \\
      --target-speaker data/speaker_references/target.wav

  # Custom test set
  python evaluate_pipeline.py --test-dir data/test_sets/my_test \\
      --streamvc-checkpoint checkpoints/streamvc/best.pt
        """
    )

    # Test set source (exactly one required)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--auto-testset", action="store_true",
        help="Auto-generate test set from LibriTTS data"
    )
    group.add_argument(
        "--test-dir", type=str, default=None,
        help="Path to custom test set directory"
    )

    # Test set options
    parser.add_argument(
        "--num-samples", type=int, default=50,
        help="Number of samples for auto test set (default: 50)"
    )
    parser.add_argument(
        "--libritts-dir", type=str,
        default="data/libritts/LibriTTS/train-clean-100",
        help="LibriTTS directory for auto test set"
    )

    # Model paths
    parser.add_argument(
        "--config", type=str, default="configs/pipeline_config.yaml",
        help="Pipeline config path"
    )
    parser.add_argument(
        "--streamvc-checkpoint", type=str, default=None,
        help="StreamVC model checkpoint (skip voice cloning eval if omitted)"
    )
    parser.add_argument(
        "--streamvc-config", type=str, default="configs/streamvc_config.yaml",
        help="StreamVC config path"
    )
    parser.add_argument(
        "--target-speaker", type=str, default=None,
        help="Target speaker reference audio for voice cloning eval"
    )

    # Runtime
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: cuda or cpu"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/eval",
        help="Directory for evaluation outputs"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for test set generation"
    )
    parser.add_argument(
        "--skip-mt", action="store_true",
        help="Skip Machine Translation (use ASR transcript directly for TTS). "
             "Useful when NLLB model is unavailable or to save RAM."
    )

    args = parser.parse_args()

    # ── Build test set ───────────────────────────────────────────────────────
    print()
    if args.auto_testset:
        print("Generating test set from LibriTTS...")
        test_samples = generate_libritts_testset(
            data_dir=args.libritts_dir,
            num_samples=args.num_samples,
            seed=args.seed
        )
    else:
        print(f"Loading custom test set from {args.test_dir}...")
        test_samples = load_custom_testset(args.test_dir)

    # ── Initialize evaluator ─────────────────────────────────────────────────
    print()
    evaluator = PipelineEvaluator(
        config_path=args.config,
        streamvc_checkpoint=args.streamvc_checkpoint,
        streamvc_config_path=args.streamvc_config,
        target_speaker_audio=args.target_speaker,
        device=args.device,
        output_dir=args.output_dir,
        skip_mt=args.skip_mt
    )

    # ── Run evaluation ───────────────────────────────────────────────────────
    report = evaluator.evaluate(test_samples)

    # Return code: 0 if no errors, 1 otherwise
    sys.exit(0 if report.get('num_errors', 0) == 0 else 1)


if __name__ == "__main__":
    main()
