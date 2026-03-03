"""
Quick test script to verify all components are working.
Run this after installation to ensure everything is set up correctly.
"""

import sys
import torch
import numpy as np


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import transformers
        import torchaudio
        import librosa
        import soundfile
        import faster_whisper
        import parselmouth
        print("✓ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_models():
    """Test that models can be loaded."""
    print("\nTesting model loading...")
    
    # Test if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    # Test a simple model load (transformers)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        print("✓ Transformers library working")
        return True
    except Exception as e:
        print(f"✗ Model loading error: {e}")
        return False


def test_audio_processing():
    """Test basic audio processing."""
    print("\nTesting audio processing...")
    
    try:
        import librosa
        import soundfile as sf
        
        # Generate test audio (sine wave)
        sr = 16000
        duration = 1.0
        freq = 440  # A4 note
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        
        # Save and load
        test_file = "test_audio.wav"
        sf.write(test_file, audio, sr)
        loaded_audio, loaded_sr = librosa.load(test_file, sr=sr)
        
        # Clean up
        import os
        os.remove(test_file)
        
        print("✓ Audio processing working")
        return True
    except Exception as e:
        print(f"✗ Audio processing error: {e}")
        return False


def test_pytorch():
    """Test PyTorch functionality."""
    print("\nTesting PyTorch...")
    
    try:
        # Test basic operations
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = x + y
        
        # Test GPU if available
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = x_gpu + y_gpu
            print(f"✓ PyTorch working (CUDA available, device: {torch.cuda.get_device_name(0)})")
        else:
            print("✓ PyTorch working (CPU only)")
        
        return True
    except Exception as e:
        print(f"✗ PyTorch error: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("VoiceCloneify Installation Test")
    print("="*60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Model Loading", test_models),
        ("Audio Processing", test_audio_processing),
        ("PyTorch", test_pytorch)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("Test Results:")
    print("="*60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:.<40} {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed! Installation is working correctly.")
        print("\nNext steps:")
        print("  1. Download datasets: python data/download_datasets.py")
        print("  2. Extract HuBERT centroids: python data/extract_hubert_centroids.py")
        print("  3. Train StreamVC: python streamvc/train.py")
        print("  4. Run pipeline: python run_pipeline.py --input audio.wav")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("\nTry reinstalling dependencies:")
        print("  pip install -r requirements.txt")
    
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
