# VoiceCloneify Pipeline Architecture & Methodology

## 1. Architectural Overview

VoiceCloneify is a **Cascaded Speech-to-Speech Translation (S2ST)** system. It does not translate "Voice A" directly to "Voice B" in one step. Instead, it deconstructs the problem into modular phases, acting like an assembly line where each station modifies the product before passing it on.

### The Data Flow
$$ \text{Audio}_{Input} \xrightarrow{ASR} \text{Text}_{Source} \xrightarrow{MT} \text{Text}_{Target} \xrightarrow{TTS} \text{Audio}_{Target}^{(Generic)} \xrightarrow{StreamVC} \text{Audio}_{Target}^{(Cloned)} $$

---

## 2. File-by-File Methodology Breakdown

### 🟢 Phase 1: The Orchestrator (`/`)

#### `run_pipeline.py`
*   **Role:** The Conductor.
*   **Methodology:**
    1.  **Model Loading:** It acts as a dependency injector, initializing the heavy classes (ASR, MT, TTS, StreamVC) once at startup to avoid reloading latency.
    2.  **Audio Ingestion:** Uses `librosa` to load audio. Crucially, it resamples everything to **16kHz**. Why? Because HuBERT and many speech models are pretrained specifically on 16kHz audio. Using 44.1kHz would break the feature extraction.
    3.  **Pipeline Chaining:** It explicitly passes the output of one function as the input to the next.
    4.  **Reference Handling:** It loads the `target_speaker` audio file, extracts the speaker embedding *once* (using StreamVC's Speaker Encoder), and reuses that embedding for every chunk of audio generated. This is efficient.

---

### 🔵 Phase 2: Speech & Text Processing (`asr/`, `mt/`, `tts/`)

#### `asr/streaming_asr.py`
*   **Role:** The Scribe (Ear).
*   **Algorithm:** **Distil-Whisper**.
*   **Methodology:**
    *   **Transformer Encoder-Decoder:** It processes raw audio waveforms directly.
    *   **Chunking:** The file implements logic to handle long audio files. Whisper has a context window (usually 30s). This script ensures audio is fed in appropriate chunks to prevent hallucination or cutoff.
    *   **Output:** Returns a raw text string (e.g., "Hello world").

#### `mt/translator.py`
*   **Role:** The Translator (Brain).
*   **Algorithm:** **NLLB-200 (No Language Left Behind)**.
*   **Methodology:**
    *   **Seq2Seq Transformer:** It uses an Encoder to read the English text into a "thought vector" and a Decoder to generate French text from that thought.
    *   **Tokenization:** It converts words into sub-word tokens (IDs).
    *   **Beam Search:** When generating translation, it explores multiple possible sentence endings to find the most statistically probable translation, not just the first word that fits.

#### `tts/text_to_speech.py`
*   **Role:** The Narrator (Mouth).
*   **Algorithm:** **SpeechT5 + HiFi-GAN Vocoder**.
*   **Methodology:**
    *   **Acoustic Modeling:** First, SpeechT5 converts the French text into a **Log-Mel Spectrogram**. This is a visual representation of the sound (time vs. frequency), but it's not sound yet.
    *   **Vocoding:** The HiFi-GAN Vocoder takes this spectrogram and generates the actual raw waveform (time vs. amplitude).
    *   **Speaker Vectors:** It uses a default "x-vector" (speaker embedding) to produce a consistent, clean, but generic voice.

---

### 🔴 Phase 3: StreamVC - The Voice Cloning Engine (`streamvc/`)

This is the custom implementation of the *StreamVC* paper. This folder is a self-contained deep learning project.

#### `streamvc/model.py`
*   **Role:** The Generator Network.
*   **Methodology:** **Disentangled Representation Learning**.
    *   **Goal:** To separate speech into $C$ (Content), $S$ (Speaker), and $P$ (Pitch).
    *   **`ContentEncoder`:**
        *   **Input:** Raw Audio (Source).
        *   **Architecture:** Causal Convolutional layers. "Causal" means it uses padding on the *left* side only.
        *   **Why?** In a live stream, you can't see the future. Standard convolutions look at $t-1$, $t$, and $t+1$. Causal ones look only at $t-2, t-1, t$. This enables real-time usage.
        *   **Task:** Predicts "Soft HuBERT Units". It tries to guess what phoneme cluster the audio belongs to.
    *   **`SpeakerEncoder`:**
        *   **Input:** Mel Spectrogram (Target Speaker).
        *   **Architecture:** Standard Convolutional network + **Global Average Pooling**.
        *   **Why?** We want to squash the *time* dimension. We don't care *when* the target speaker spoke; we just want the *average texture* of their voice. The output is a single vector (e.g., 256 numbers) representing "The Essence of User".
    *   **`StreamDecoder`:**
        *   **Input:** Content Features + Pitch Features.
        *   **Conditioning:** Uses **FiLM (Feature-wise Linear Modulation)**.
        *   **Mechanism:** The FiLM layer takes the Speaker Vector and learns a `Scale` ($\gamma$) and `Shift` ($\beta$) for the neural network activations.
        *   **Analogy:** It's like applying an Instagram filter (Speaker Vector) over a photo (Content).
        *   **Output:** The final cloned waveform.

#### `streamvc/f0_extractor.py`
*   **Role:** Pitch Tracker.
*   **Methodology:** **Yin Algorithm / PyWorld**.
    *   **Whitening:** It calculates the mean ($\mu$) and standard deviation ($\sigma$) of the pitch.
    *   **Normalization:** $F0_{norm} = \frac{F0 - \mu}{\sigma}$.
    *   **Why?** If the source is a high-pitched child and the target is a deep-voiced adult, we can't just copy the pitch. We normalize the child's pitch to "zero center" and then potentially denormalize it or let the model re-inject pitch dynamics suitable for the target.

#### `streamvc/discriminator.py`
*   **Role:** The Critic (Adversary).
*   **Methodology:** **Multi-Scale Discriminator (MSD)**.
    *   **Problem:** Audio has structure at different scales. High frequency (timbre/fuzziness) and Low frequency (prosody/rhythm).
    *   **Solution:** We create 3 separate discriminators:
        1.  analyzes raw audio (1x).
        2.  analyzes downsampled audio (0.5x).
        3.  analyzes heavily downsampled audio (0.25x).
    *   **Feature Matching Loss:** The discriminator doesn't just say "True/False". It returns the *internal features* of its layers. The Generator is forced to match these internal statistics, which stabilizes training massively.

#### `streamvc/train.py`
*   **Role:** The Gym.
*   **Methodology:**
    *   **Loss Landscape:** It optimizes a complex sum of losses:
        $$ L_{total} = \lambda_{rec}L_{1} + \lambda_{content}L_{hubert} + \lambda_{adv}L_{GAN} + \lambda_{feat}L_{FM} $$
    *   **HuBERT Targets:** It uses the pre-computed centroids (from `data/`) as the "Correct Answer" for the Content Encoder.

---

### 🟡 Phase 4: Data Engineering (`data/`)

#### `data/extract_hubert_centroids.py`
*   **Role:** The Teacher Creator.
*   **Methodology:** **K-Means Clustering on Self-Supervised Features**.
    1.  **Inference:** Runs `hubert-base-ls960` on the dataset. HuBERT outputs a vector (size 768) for every 20ms of audio.
    2.  **Collection:** Collects millions of these vectors.
    3.  **Clustering:** Runs K-Means to find 100 "Centroids" (representative centers).
    4.  **Result:** These 100 centers represent the 100 most distinct sounds in the English language (roughly mapping to phonemes/allophones).
    5.  **Storage:** Saves to `.pkl` file. This file is the "dictionary" the StreamVC model learns to read.

#### `data/download_datasets.py`
*   **Role:** The Librarian.
*   **Methodology:**
    *   Automates the fetching of standard speech datasets (LibriTTS, VCTK, LJSpeech).
    *   Handles unzipping and folder restructuring so the `dataset.py` loader can find them easily.

---

## 3. Configuration Management (`configs/`)

#### `configs/streamvc_config.yaml`
*   **Role:** The Blueprint.
*   **Why YAML?** Allows changing experiment parameters (learning rate, model size) without changing code.
*   **Key Params:**
    *   `hop_length`: Defines the time resolution. Lower = higher quality but more computation.
    *   `lookahead_frames`: Defines latency. `2` frames = ~40ms latency.
