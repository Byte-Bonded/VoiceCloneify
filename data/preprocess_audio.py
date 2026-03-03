"""
Preprocess audio datasets to consistent 16kHz mono format.
"""

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List

import librosa
import soundfile as sf
from tqdm import tqdm


def process_audio_file(args):
    """Process a single audio file to 16kHz mono."""
    input_path, output_path, target_sr = args
    
    try:
        # Load and resample
        audio, sr = librosa.load(input_path, sr=target_sr, mono=True)
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, target_sr)
        
        return True, None
    except Exception as e:
        return False, str(e)


def preprocess_dataset(
    input_dir: Path,
    output_dir: Path,
    extensions: List[str] = [".wav", ".flac"],
    target_sr: int = 16000,
    num_workers: int = 4
):
    """Preprocess all audio files in a directory."""
    
    # Find all audio files
    audio_files = []
    for ext in extensions:
        audio_files.extend(input_dir.rglob(f"*{ext}"))
    
    print(f"Found {len(audio_files)} audio files in {input_dir}")
    
    # Prepare processing tasks
    tasks = []
    for input_path in audio_files:
        rel_path = input_path.relative_to(input_dir)
        output_path = output_dir / rel_path.with_suffix('.wav')
        
        # Skip if already processed
        if output_path.exists():
            continue
            
        tasks.append((input_path, output_path, target_sr))
    
    if not tasks:
        print("All files already processed!")
        return
    
    print(f"Processing {len(tasks)} files with {num_workers} workers...")
    
    # Process in parallel
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_audio_file, tasks),
            total=len(tasks),
            desc="Processing audio"
        ))
    
    # Report results
    success_count = sum(1 for success, _ in results if success)
    print(f"✓ Successfully processed {success_count}/{len(tasks)} files")
    
    # Report errors
    errors = [(i, err) for i, (success, err) in enumerate(results) if not success]
    if errors:
        print(f"⚠ {len(errors)} files failed:")
        for i, err in errors[:5]:  # Show first 5 errors
            print(f"  - {tasks[i][0]}: {err}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess audio datasets to 16kHz mono"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory with audio files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for processed files"
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=16000,
        help="Target sample rate (default: 16000)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    preprocess_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        target_sr=args.target_sr,
        num_workers=args.num_workers
    )
    
    print(f"\n✅ Preprocessing complete! Output: {output_dir}")


if __name__ == "__main__":
    main()
