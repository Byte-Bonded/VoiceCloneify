"""
Data download and preparation scripts for VoiceCloneify pipeline.
Downloads LibriTTS, VCTK, and LJSpeech datasets.
"""

import argparse
import os
import tarfile
import zipfile
import time
from pathlib import Path
from urllib.request import urlretrieve

import tqdm


class DownloadProgressBar(tqdm.tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: Path, max_retries: int = 3):
    """Download file with progress bar and retry logic."""
    for attempt in range(max_retries):
        try:
            with DownloadProgressBar(
                unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]
            ) as t:
                urlretrieve(url, filename=output_path, reporthook=t.update_to)
            return  # Success
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10
                print(f"\n⚠ Download failed: {e}")
                print(f"Retrying in {wait_time}s... (attempt {attempt + 2}/{max_retries})")
                time.sleep(wait_time)
                # Remove partial file
                if os.path.exists(output_path):
                    os.remove(output_path)
            else:
                print(f"\n✗ Failed after {max_retries} attempts")
                print(f"You can download manually from: {url}")
                raise


def download_libritts(data_dir: Path):
    """Download LibriTTS train-clean-100 (~6GB)."""
    print("\n[1/3] Downloading LibriTTS train-clean-100...")
    
    libritts_dir = data_dir / "libritts"
    libritts_dir.mkdir(exist_ok=True, parents=True)
    
    url = "https://www.openslr.org/resources/60/train-clean-100.tar.gz"
    output_file = libritts_dir / "train-clean-100.tar.gz"
    
    if not output_file.exists():
        download_url(url, output_file)
        print(f"✓ Downloaded to {output_file}")
    else:
        print(f"✓ Already exists: {output_file}")
    
    # Extract
    extract_dir = libritts_dir / "LibriTTS"
    if not extract_dir.exists():
        print("Extracting archive...")
        with tarfile.open(output_file, "r:gz") as tar:
            tar.extractall(path=libritts_dir)
        print(f"✓ Extracted to {extract_dir}")
    else:
        print(f"✓ Already extracted: {extract_dir}")


def download_ljspeech(data_dir: Path):
    """Download LJSpeech (~2.6GB)."""
    print("\n[2/3] Downloading LJSpeech...")
    
    ljspeech_dir = data_dir / "ljspeech"
    ljspeech_dir.mkdir(exist_ok=True, parents=True)
    
    url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    output_file = ljspeech_dir / "LJSpeech-1.1.tar.bz2"
    
    if not output_file.exists():
        download_url(url, output_file)
        print(f"✓ Downloaded to {output_file}")
    else:
        print(f"✓ Already exists: {output_file}")
    
    # Extract
    extract_dir = ljspeech_dir / "LJSpeech-1.1"
    if not extract_dir.exists():
        print("Extracting archive...")
        with tarfile.open(output_file, "r:bz2") as tar:
            tar.extractall(path=ljspeech_dir)
        print(f"✓ Extracted to {extract_dir}")
    else:
        print(f"✓ Already extracted: {extract_dir}")


def download_vctk(data_dir: Path):
    """Download VCTK (~11GB)."""
    print("\n[3/3] Downloading VCTK...")
    
    vctk_dir = data_dir / "vctk"
    vctk_dir.mkdir(exist_ok=True, parents=True)
    
    url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
    output_file = vctk_dir / "VCTK-Corpus-0.92.zip"
    
    if not output_file.exists():
        download_url(url, output_file)
        print(f"✓ Downloaded to {output_file}")
    else:
        print(f"✓ Already exists: {output_file}")
    
    # Extract
    extract_dir = vctk_dir / "VCTK-Corpus-0.92"
    if not extract_dir.exists():
        print("Extracting archive... (this may take several minutes)")
        with zipfile.ZipFile(output_file, 'r') as zip_ref:
            zip_ref.extractall(vctk_dir)
        print(f"✓ Extracted to {extract_dir}")
    else:
        print(f"✓ Already extracted: {extract_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for VoiceCloneify"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to download datasets (default: ./data)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["libritts", "ljspeech", "vctk", "all"],
        default=["all"],
        help="Datasets to download (default: all)"
    )
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(exist_ok=True, parents=True)
    
    datasets = args.datasets
    if "all" in datasets:
        datasets = ["libritts", "ljspeech", "vctk"]
    
    print(f"📥 Downloading datasets to: {data_dir}")
    print(f"Selected datasets: {', '.join(datasets)}")
    
    if "libritts" in datasets:
        download_libritts(data_dir)
    
    if "ljspeech" in datasets:
        download_ljspeech(data_dir)
    
    if "vctk" in datasets:
        download_vctk(data_dir)
    
    print("\n✅ All downloads complete!")
    print(f"\nDataset locations:")
    print(f"  LibriTTS: {data_dir / 'libritts' / 'LibriTTS' / 'train-clean-100'}")
    print(f"  LJSpeech: {data_dir / 'ljspeech' / 'LJSpeech-1.1'}")
    print(f"  VCTK:     {data_dir / 'vctk' / 'VCTK-Corpus-0.92'}")


if __name__ == "__main__":
    main()
