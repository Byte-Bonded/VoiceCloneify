#!/usr/bin/env python3
"""
Quick installation test - tests core imports only
"""

import sys

def main():
    print("=" * 60)
    print("Quick Installation Test")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        tests_passed += 1
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        tests_failed += 1
    
    # Test 2: Transformers
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
        tests_passed += 1
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        tests_failed += 1
    
    # Test 3: Torchaudio
    try:
        import torchaudio
        print(f"✓ Torchaudio {torchaudio.__version__}")
        tests_passed += 1
    except ImportError as e:
        print(f"✗ Torchaudio import failed: {e}")
        tests_failed += 1
    
    # Test 4: Librosa
    try:
        import librosa
        print(f"✓ Librosa {librosa.__version__}")
        tests_passed += 1
    except ImportError as e:
        print(f"✗ Librosa import failed: {e}")
        tests_failed += 1
    
    # Test 5: Soundfile
    try:
        import soundfile
        print(f"✓ Soundfile {soundfile.__version__}")
        tests_passed += 1
    except ImportError as e:
        print(f"✗ Soundfile import failed: {e}")
        tests_failed += 1
    
    # Test 6: Faster Whisper
    try:
        import faster_whisper
        print(f"✓ Faster Whisper")
        tests_passed += 1
    except ImportError as e:
        print(f"✗ Faster Whisper import failed: {e}")
        tests_failed += 1
    
    print("=" * 60)
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)
    
    if tests_failed > 0:
        print("\nTo install missing packages:")
        print("  source venv/bin/activate")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("\n✓ All core packages installed correctly!")
        print("\nNext steps:")
        print("  1. Download datasets: python data/download_datasets.py")
        print("  2. Run pipeline: python run_pipeline.py --input audio.wav")
        sys.exit(0)

if __name__ == "__main__":
    main()
