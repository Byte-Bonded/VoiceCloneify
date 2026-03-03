"""
Standalone StreamVC Voice Conversion Evaluator
===============================================
Evaluates StreamVC voice conversion quality directly on LibriTTS audio pairs.
Skips ASR/MT/TTS — focuses on the core VC metrics only.

Metrics:
  - MCD (Mel Cepstral Distortion) [lower is better, < 8 dB = good]
  - F0 Pearson Correlation [higher is better, > 0.9 = good]
  - F0 RMSE [lower is better]
  - Speaker Cosine Similarity via StreamVC encoder [higher = better, > 0.85]
  - Energy Correlation [higher is better]
  - SNR [higher is better]
  - RTF (Real-Time Factor) [< 1.0 = real-time capable]

Usage:
  python eval_vc_only.py --checkpoint checkpoints/streamvc/best_model.pt \
      --num-samples 20 --device cuda
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
import librosa
import soundfile as sf
from scipy.stats import pearsonr


# ── Audio utilities ────────────────────────────────────────────────────────────

def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def compute_mcd(ref: np.ndarray, gen: np.ndarray, sr: int = 16000,
                n_mfcc: int = 13) -> float:
    """Mel Cepstral Distortion in dB (C0 excluded per standard MCD definition)."""
    min_len = min(len(ref), len(gen))
    if min_len < 256:
        return float('nan')
    ref = ref[:min_len]
    gen = gen[:min_len]
    mfcc_ref = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=n_mfcc)
    mfcc_gen = librosa.feature.mfcc(y=gen, sr=sr, n_mfcc=n_mfcc)
    # Align frame counts
    min_frames = min(mfcc_ref.shape[1], mfcc_gen.shape[1])
    mfcc_ref = mfcc_ref[:, :min_frames]
    mfcc_gen = mfcc_gen[:, :min_frames]
    # Exclude C0 (energy coefficient) — standard MCD uses C1..C12
    diff = mfcc_ref[1:] - mfcc_gen[1:]
    mcd = (10.0 / np.log(10.0)) * np.sqrt(2.0) * np.mean(
        np.sqrt(np.sum(diff ** 2, axis=0))
    )
    return float(mcd)


def extract_f0(audio: np.ndarray, sr: int = 16000,
               f0_min: float = 50.0, f0_max: float = 600.0) -> np.ndarray:
    """Extract F0 using librosa pyin."""
    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=f0_min, fmax=f0_max, sr=sr,
        frame_length=2048, hop_length=256
    )
    return np.nan_to_num(f0, nan=0.0)


def compute_f0_metrics(ref_f0: np.ndarray, gen_f0: np.ndarray):
    """Compute F0 correlation and RMSE on voiced frames."""
    # Align lengths
    min_len = min(len(ref_f0), len(gen_f0))
    ref_f0 = ref_f0[:min_len]
    gen_f0 = gen_f0[:min_len]
    # Only voiced frames (both non-zero)
    voiced = (ref_f0 > 0) & (gen_f0 > 0)
    if voiced.sum() < 10:
        return float('nan'), float('nan')
    r = pearsonr(ref_f0[voiced], gen_f0[voiced])[0]
    rmse = float(np.sqrt(np.mean((ref_f0[voiced] - gen_f0[voiced]) ** 2)))
    return float(r), rmse


def compute_snr(audio: np.ndarray) -> float:
    """Signal-to-noise ratio estimate."""
    signal_power = np.mean(audio ** 2)
    if signal_power < 1e-10:
        return -100.0
    noise_floor = np.percentile(np.abs(audio), 5) ** 2
    if noise_floor < 1e-12:
        return 60.0
    return float(10 * np.log10(signal_power / noise_floor))


def compute_energy_corr(ref: np.ndarray, gen: np.ndarray,
                         frame_length: int = 512, hop_length: int = 256) -> float:
    ref_e = librosa.feature.rms(y=ref, frame_length=frame_length,
                                 hop_length=hop_length)[0]
    gen_e = librosa.feature.rms(y=gen, frame_length=frame_length,
                                 hop_length=hop_length)[0]
    min_f = min(len(ref_e), len(gen_e))
    if min_f < 2:
        return float('nan')
    return float(pearsonr(ref_e[:min_f], gen_e[:min_f])[0])


# ── Dataset builder ───────────────────────────────────────────────────────────

def gather_libritts_samples(libritts_dir: str, num_samples: int,
                              seed: int = 42, min_dur: float = 2.0,
                              max_dur: float = 10.0) -> list:
    """Return list of (source_wav, reference_wav, speaker_id) triples."""
    rng = random.Random(seed)
    root = Path(libritts_dir)
    wav_files = sorted(root.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {libritts_dir}")

    # Group by speaker
    by_speaker: dict = {}
    for f in wav_files:
        spk = f.parent.parent.name  # LibriTTS: root/spk/chapter/file.wav
        by_speaker.setdefault(spk, []).append(f)

    # Build pairs: different utterances, same speaker as reference
    # (to measure: does converted output sound like reference speaker?)
    pairs = []
    speakers = list(by_speaker.keys())
    rng.shuffle(speakers)

    for spk in speakers:
        utts = by_speaker[spk]
        if len(utts) < 2:
            continue
        rng.shuffle(utts)
        # source = utt[0], reference = utt[1]
        src, ref = utts[0], utts[1]
        pairs.append({'source': str(src), 'reference': str(ref),
                      'speaker': spk})
        if len(pairs) >= num_samples:
            break

    # If not enough speakers, allow same-speaker repeats with different utts
    if len(pairs) < num_samples:
        for spk in speakers:
            utts = by_speaker[spk]
            for i in range(0, len(utts) - 1, 2):
                pairs.append({'source': str(utts[i]), 'reference': str(utts[i + 1]),
                              'speaker': spk})
                if len(pairs) >= num_samples:
                    break
            if len(pairs) >= num_samples:
                break

    rng.shuffle(pairs)
    return pairs[:num_samples]


# ── Evaluator ─────────────────────────────────────────────────────────────────

class VCEvaluator:
    def __init__(self, checkpoint: str, config: str,
                 device: str = "cuda", output_dir: str = "outputs/vc_eval"):
        self.sr = 16000
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print(f"Device: {self.device}")

        # Load StreamVC
        print(f"Loading StreamVC from {checkpoint} ...")
        with open(config) as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg

        from streamvc.model import StreamVC
        from streamvc.audio_utils import MelSpectrogramExtractor
        from streamvc.f0_extractor import F0ExtractorModule

        self.model = StreamVC(cfg).to(self.device)
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        print(f"  Checkpoint epoch: {ckpt.get('epoch', '?')}")
        print(f"  Checkpoint val_loss: {ckpt.get('val_loss', '?')}")

        audio_cfg = cfg['audio']
        self.mel_ext = MelSpectrogramExtractor(
            sample_rate=audio_cfg['sample_rate'],
            n_fft=audio_cfg['n_fft'],
            hop_length=audio_cfg['hop_length'],
            n_mels=audio_cfg['n_mels']
        ).to(self.device)

        f0_cfg = cfg['model']['f0_extractor']
        self._f0_module = F0ExtractorModule(
            sample_rate=audio_cfg['sample_rate'],
            frame_length=f0_cfg['frame_length'],
            f0_min=f0_cfg['f0_min'],
            f0_max=f0_cfg['f0_max'],
            whitening=f0_cfg['whitening']
        )

        print("✓ StreamVC loaded")

        self._embed_fn = self._streamvc_embed

    @torch.no_grad()
    def _streamvc_embed(self, audio_np: np.ndarray) -> np.ndarray:
        """Get speaker embedding from StreamVC speaker encoder mean pool."""
        audio_t = torch.tensor(audio_np).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,T)
        mel = self.mel_ext(audio_t)  # (1, n_mels, T_mel)
        # Use speaker encoder to get fixed-size embedding
        emb = self.model.speaker_encoder(mel)  # (1, emb_dim)
        emb_np = emb.squeeze(0).cpu().numpy()
        return emb_np / (np.linalg.norm(emb_np) + 1e-8)

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    @torch.no_grad()
    def voice_convert(self, source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Run voice conversion: source content + reference speaker → output audio."""
        # (B=1, 1, T)
        src_t = torch.tensor(source).unsqueeze(0).unsqueeze(0).to(self.device)
        ref_t = torch.tensor(reference).unsqueeze(0).unsqueeze(0).to(self.device)

        # Target mel from reference speaker — use ALL frames
        # (speaker encoder uses attention pooling → single embedding regardless of length)
        ref_mel = self.mel_ext(ref_t)  # (1, n_mels, T_mel)

        # F0 features for source
        f0_features = self._f0_module(src_t)  # (1, 9, T_f0)

        # Do NOT align ref_mel with f0_features — they are from different audio clips.
        # The speaker encoder pools ref_mel to a single embedding.
        # The decoder internally aligns content_features with f0_features.

        # Forward pass → output audio (1, 1, T_out)
        output_audio, _ = self.model(src_t, ref_mel, f0_features)
        return output_audio.squeeze().cpu().numpy()

    def evaluate_sample(self, idx: int, sample: dict) -> dict:
        result = {'idx': idx, 'speaker': sample['speaker'],
                  'source': sample['source'], 'reference': sample['reference']}

        src_audio = load_audio(sample['source'], self.sr)
        ref_audio = load_audio(sample['reference'], self.sr)

        # Run voice conversion — returns audio waveform directly
        t0 = time.time()
        try:
            vc_audio = self.voice_convert(src_audio, ref_audio)
            result['vc_time_s'] = time.time() - t0
            result['rtf'] = result['vc_time_s'] / (len(src_audio) / self.sr)
            result['vc_success'] = True
        except Exception as e:
            result['error'] = str(e)
            result['vc_success'] = False
            print(f"  [Sample {idx}] VC FAILED: {e}")
            return result

        # Save output audio (direct waveform from model)
        out_path = self.output_dir / f"sample_{idx:04d}_vc.wav"
        sf.write(str(out_path), vc_audio, self.sr)
        result['output_path'] = str(out_path)

        # MCD: source vs converted
        result['mcd_src_vc'] = compute_mcd(src_audio, vc_audio, self.sr)

        # F0 metrics: compare source F0 to converted F0
        src_f0 = extract_f0(src_audio, self.sr)
        vc_f0 = extract_f0(vc_audio, self.sr)
        f0_corr, f0_rmse = compute_f0_metrics(src_f0, vc_f0)
        result['f0_pearson'] = f0_corr
        result['f0_rmse_hz'] = f0_rmse

        # Energy correlation
        result['energy_corr'] = compute_energy_corr(src_audio, vc_audio)

        # SNR of output
        result['vc_snr_db'] = compute_snr(vc_audio)

        # Speaker similarity: does VC output sound like reference speaker?
        try:
            ref_emb = self._embed_fn(ref_audio)
            vc_emb = self._embed_fn(vc_audio)
            src_emb = self._embed_fn(src_audio)
            result['spk_sim_vc_ref'] = self._cosine_sim(vc_emb, ref_emb)
            result['spk_sim_src_ref'] = self._cosine_sim(src_emb, ref_emb)
            result['spk_sim_delta'] = (result['spk_sim_vc_ref']
                                       - result['spk_sim_src_ref'])
        except Exception as e:
            print(f"  [Sample {idx}] Speaker sim failed: {e}")

        return result

    def evaluate(self, samples: list) -> dict:
        results = []
        n = len(samples)
        print(f"\nEvaluating {n} samples...")
        print("-" * 50)

        for i, sample in enumerate(samples):
            print(f"  [{i+1}/{n}] Speaker {sample['speaker']}: "
                  f"{Path(sample['source']).name}", flush=True)
            r = self.evaluate_sample(i, sample)
            results.append(r)

            # Print per-sample summary
            if r.get('vc_success'):
                spk_sim = r.get('spk_sim_vc_ref', float('nan'))
                mcd = r.get('mcd_src_vc', float('nan'))
                f0c = r.get('f0_pearson', float('nan'))
                rtf = r.get('rtf', float('nan'))
                print(f"    MCD={mcd:.2f}dB  SpkSim(vc↔ref)={spk_sim:.4f}  "
                      f"F0Corr={f0c:.4f}  RTF={rtf:.3f}x")

        # Aggregate
        def safe_mean(key):
            vals = [r[key] for r in results if key in r and not np.isnan(r[key])]
            return float(np.mean(vals)) if vals else float('nan')

        report = {
            'num_samples': n,
            'num_success': sum(1 for r in results if r.get('vc_success')),
            'num_errors': sum(1 for r in results if not r.get('vc_success')),
            'metrics': {
                'mcd_src_vc_db':    safe_mean('mcd_src_vc'),
                'f0_pearson':        safe_mean('f0_pearson'),
                'f0_rmse_hz':        safe_mean('f0_rmse_hz'),
                'energy_corr':       safe_mean('energy_corr'),
                'vc_snr_db':         safe_mean('vc_snr_db'),
                'spk_sim_vc_ref':    safe_mean('spk_sim_vc_ref'),
                'spk_sim_src_ref':   safe_mean('spk_sim_src_ref'),
                'spk_sim_delta':     safe_mean('spk_sim_delta'),
                'rtf':               safe_mean('rtf'),
            },
            'per_sample': results
        }

        # Save report
        report_path = self.output_dir / "vc_eval_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved to {report_path}")

        # Print summary
        m = report['metrics']
        print("\n" + "=" * 60)
        print("  VOICE CONVERSION EVALUATION SUMMARY")
        print("=" * 60)
        print(f"  Samples:          {report['num_success']}/{n} succeeded")
        print(f"  MCD (src→vc):      {m['mcd_src_vc_db']:.2f} dB   "
              f"[good < 8 dB]")
        print(f"  F0 Pearson:        {m['f0_pearson']:.4f}   "
              f"[good > 0.9]")
        print(f"  F0 RMSE:           {m['f0_rmse_hz']:.1f} Hz")
        print(f"  Energy Corr:       {m['energy_corr']:.4f}")
        print(f"  Output SNR:        {m['vc_snr_db']:.1f} dB")
        print(f"  SpkSim vc↔ref:     {m['spk_sim_vc_ref']:.4f}   "
              f"[good > 0.85]")
        print(f"  SpkSim src↔ref:    {m['spk_sim_src_ref']:.4f}   "
              f"(baseline, same spk)")
        print(f"  SpkSim delta:      {m['spk_sim_delta']:+.4f}   "
              f"[positive = vc helped]")
        print(f"  RTF:               {m['rtf']:.3f}x   "
              f"[< 1.0 = real-time]")
        print("=" * 60)

        # Decision guidance
        mcd = m['mcd_src_vc_db']
        sim = m['spk_sim_vc_ref']
        print("\nDECISION GUIDANCE:")
        if not np.isnan(mcd) and not np.isnan(sim):
            if mcd < 8 and sim > 0.85:
                print("  ✅ GOOD — Model produces high quality voice conversions.")
                print("     best_model.pt is ready for use.")
            elif mcd < 15 and sim > 0.70:
                print("  ⚠️  MARGINAL — Acceptable quality but room for improvement.")
                print("     Consider: fix discriminator collapse and train 20 more epochs.")
            else:
                print("  ❌ POOR — Voice conversion quality is insufficient.")
                print("     Recommended: reset discriminator, retrain from epoch 28 checkpoint.")
        print()

        return report


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate StreamVC voice conversion quality directly"
    )
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/streamvc/best_model.pt",
                        help="Path to StreamVC checkpoint")
    parser.add_argument("--config", type=str,
                        default="configs/streamvc_config.yaml",
                        help="Path to StreamVC config")
    parser.add_argument("--libritts-dir", type=str,
                        default="data/libritts/LibriTTS/train-clean-100",
                        help="LibriTTS directory")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Number of audio pairs to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/vc_eval")
    args = parser.parse_args()

    print(f"\nStreamVC Voice Conversion Evaluator")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print()

    samples = gather_libritts_samples(
        args.libritts_dir, args.num_samples, args.seed
    )
    print(f"Gathered {len(samples)} audio pairs from LibriTTS")

    evaluator = VCEvaluator(
        checkpoint=args.checkpoint,
        config=args.config,
        device=args.device,
        output_dir=args.output_dir
    )

    evaluator.evaluate(samples)


if __name__ == "__main__":
    main()
