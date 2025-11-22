# ML-Based Pitch Detection Feature Investigation

## Executive Summary

The ML feature in PitchVis is an experimental machine learning-based approach to fundamental frequency (pitch) detection using neural networks. The system uses Variable Q Transform (VQT) spectrograms as input to predict active MIDI pitches in polyphonic audio. According to the project history, this feature was "not particularly successful" and represents an early attempt at integrating deep learning into the pitch detection pipeline.

**Key Findings:**
- Uses a simple CNN+MLP hybrid architecture with ~1M parameters
- Trained on synthesized MIDI data (primarily Bach compositions) using SoundFont rendering
- Employs VQT with temporal history (5 frames) as input features
- Multi-label binary classification task (128 MIDI pitches)
- Currently feature-gated and not actively used in production

---

## Current Implementation

### Architecture Overview

The ML system consists of three main components:

1. **Training Pipeline** (`pitchvis_train/`)
   - Rust-based data generation: `src/train.rs`
   - Python-based model training: `train.py`
   - Outputs: `data.npy` (training data), `model.pt` (TorchScript model)

2. **Model Architecture** (PyTorch)
   - Input: VQT with temporal history (5 frames × 252 bins = 1,260 features)
   - 1D Convolution layer: 1→16 channels, kernel=5, stride=2
   - Max pooling: kernel=2
   - Fully connected layers: 2 hidden layers of 1,024 neurons
   - Dropout: 0.1
   - Output: 128 binary predictions (one per MIDI pitch)
   - Activation: Sigmoid (BCE loss)

3. **Inference System** (`pitchvis_viewer/src/ml_system.rs`)
   - Loads TorchScript model via `tch-rs`
   - Runs inference on CUDA (GPU-only)
   - Applies 0.5 threshold to probabilities
   - Stores results in `AnalysisState::ml_midi_base_pitches`

### Data Generation Pipeline

#### VQT Parameters
```rust
SR: 22,050 Hz
N_FFT: 32,768 samples
FREQ_A1: 55 Hz (MIDI 33)
BUCKETS_PER_OCTAVE: 36
OCTAVES: 7
Q: 10.0
GAMMA: 53.0
```

#### Training Data Synthesis Process

1. **MIDI Processing**
   - Reads MIDI files from `midi/` directory (primarily Bach compositions)
   - Uses `MuseScore_General.sf2` SoundFont for synthesis
   - Sample rate: 22,050 Hz

2. **Audio Synthesis**
   - Uses `rustysynth` to render MIDI → audio
   - Downmixes stereo to mono
   - Applies dynamic AGC (Automatic Gain Control)
   - Skips silent regions to avoid biasing the model

3. **VQT Analysis**
   - Processes audio in chunks aligned with VQT delay (~100ms)
   - Captures active MIDI notes at each time step
   - Associates note velocities/gains with VQT features
   - Step size: 3 chunks (downsampled temporal resolution)

4. **Data Format**
   - Input: 252 VQT bins (7 octaves × 36 bins/octave)
   - Target: 128-dimensional binary vector (MIDI 0-127)
   - Only notes with gain > 0.5 are marked as active
   - Saved as NumPy array: `data.npy`

#### Temporal Windowing (Python)

The training script creates 5-frame sliding windows:
- Flattens to 1,260 features (5 × 252)
- Aligns with MIDI targets from the last frame (time T)
- Train/test split: 80/20

### Model Training

#### Hyperparameters
```python
Epochs: 32
Batch size: 300
Optimizer: Adam
  - Learning rate: 1e-5
  - Weight decay: 5e-4
  - Betas: (0.9, 0.999)
  - Epsilon: 1.19e-7
Loss: Binary Cross-Entropy (BCE)
```

#### Training Process
- GPU acceleration (CUDA if available)
- No data augmentation
- No learning rate scheduling
- Evaluation metrics: F1-score (micro), accuracy

#### Model Export
- Traced to TorchScript using `torch.jit.trace`
- Saved as `model.pt`
- Deployable via `tch-rs` in Rust

### Integration with PitchVis

The ML system is integrated as an optional feature (`#[cfg(feature = "ml")]`):

```rust
// In pitchvis_viewer/src/app/desktop_app.rs
#[cfg(feature = "ml")]
let ml_model_resource =
    crate::ml_system::MlModelResource(crate::ml_system::MlModel::new("model.pt"));

#[cfg(feature = "ml")]
app.add_systems(
    Update,
    (
        update_ml_system.after(update_analysis_state_system),
        update_display_system.after(update_ml_system),
    ),
);
```

#### Inference Flow

1. Retrieves last 5 VQT frames from `AnalysisState.history`
2. Flattens to 1,260-feature vector
3. Runs GPU inference via `tch-rs`
4. Applies 0.5 threshold to get binary predictions
5. Stores results in `ml_midi_base_pitches[128]`
6. Display system can optionally visualize these predictions

**Current Usage:** The `ml_midi_base_pitches` field is stored but only minimally used in the display system (line 308 in `display_system/update.rs`).

---

## Identified Limitations

### 1. **Model Simplicity**
- Architecture is quite basic (1 conv layer + 2 FC layers)
- No temporal modeling beyond fixed 5-frame window
- No attention mechanisms to focus on relevant frequencies
- No recurrence to model longer-term musical context

### 2. **Training Data Limitations**
- Limited to one SoundFont (MuseScore_General)
- Primarily Bach MIDI files (limited musical diversity)
- No pitch variations (bends, vibrato) in training data
- No tuning variations (always A440)
- Synthesized audio lacks real-world acoustic complexity

### 3. **Feature Representation**
- Fixed 5-frame window (inflexible temporal context)
- VQT bins are concatenated, not spatially structured
- No explicit harmonic structure encoding
- Loses 2D time-frequency structure by flattening

### 4. **Training Procedure**
- No data augmentation (pitch shifts, time stretches, noise)
- No validation set (only train/test split)
- No early stopping or model checkpointing
- Hyperparameters not tuned systematically

### 5. **Inference Constraints**
- **CUDA-only**: Hardcoded `Device::Cuda(0)` prevents CPU usage
- No batch inference (processes one frame at a time)
- No temporal smoothing of predictions
- Binary threshold (0.5) is not tuned

### 6. **Evaluation Gaps**
- F1 and accuracy are coarse metrics for polyphonic pitch
- No per-pitch precision/recall analysis
- No evaluation on real audio (only synthesized test set)
- No comparison to existing pitch detection methods

---

## Literature Review: State-of-the-Art Approaches

### Modern Neural Architectures for Pitch Detection

#### 1. **CREPE (2018)** - Time-Domain CNN
- **Input:** Raw audio waveform (no manual features)
- **Architecture:** 6-layer CNN with increasing filter sizes
- **Output:** 360-bin pitch classification (20 cents resolution)
- **Performance:** State-of-the-art monophonic pitch tracking
- **Advantage:** End-to-end learning avoids hand-crafted features

**Relevance:** CREPE demonstrates that deep CNNs can learn pitch directly from raw audio, potentially outperforming traditional VQT-based approaches.

#### 2. **SwiftF0 (2024)** - Lightweight Monophonic Pitch Estimator
- **Parameters:** Only 95,842 (vs. CREPE's ~2M)
- **Speed:** 42× faster than CREPE on CPU
- **Performance:** 91.80% F1 at 10dB SNR (12pp better than CREPE)
- **Key Innovation:** Efficient architecture for resource-constrained devices

**Relevance:** Shows that lightweight models can outperform larger ones with better architecture design.

#### 3. **Deep Layered Learning for Polyphonic Pitch (2018)**
- **Approach:** Cascaded neural networks on CQT spectrograms
- **Architecture:** Multiple CNNs in series for denoising and detection
- **Innovation:** Multi-stage refinement improves polyphonic accuracy

**Relevance:** Demonstrates value of multi-stage processing and CQT/VQT representations.

#### 4. **Convolutional Recurrent Networks (CRNNs)**
- **Architecture:** CNN feature extraction + RNN temporal modeling
- **Input:** CQT spectrograms
- **Advantage:** Captures both spectral and temporal dependencies
- **Application:** Widely used for multi-pitch detection in MIREX challenges

**Relevance:** Temporal modeling via RNNs/LSTMs is crucial for music analysis.

#### 5. **Transformer-Based Approaches (2024)**

##### Audio-Visual Piano Transcription (2024)
- **Innovation:** Frequency-domain sparse attention for harmonic relationships
- **Approach:** Combines audio and visual modalities
- **Key Insight:** Attention mechanisms can model pitch relationships explicitly

##### Self-Supervised Multi-Pitch Estimation (2024)
- **Training:** Fully self-supervised (no labeled data required)
- **Advantage:** Scalable to unlimited unlabeled audio
- **Relevance:** Could reduce dependency on synthesized MIDI data

#### 6. **DeepF0 - Dilated Convolutions**
- **Key Feature:** Dilated convolutional blocks for large receptive fields
- **Advantage:** Exponential receptive field growth without parameter explosion
- **Application:** Handles longer temporal contexts efficiently

**Relevance:** Provides a way to capture longer musical context than fixed windows.

### Key Insights from Literature

1. **Raw Audio Input:** Modern models (CREPE, SwiftF0) often work on raw waveforms, learning features end-to-end.

2. **Temporal Modeling:** RNNs, LSTMs, and Transformers significantly outperform fixed-window approaches for music.

3. **Attention Mechanisms:** Can model harmonic relationships and long-range dependencies in frequency/time.

4. **Multi-Stage Refinement:** Cascaded networks (denoising → detection → refinement) improve results.

5. **Data Augmentation:** Pitch shifts, time stretches, noise injection are essential for generalization.

6. **Self-Supervised Learning:** Reduces reliance on labeled/synthesized data.

---

## Alternative Approaches

### Option 1: Enhanced VQT-Based Model (Evolutionary)

**Improvements to Current System:**

1. **Better Architecture**
   - Add 2D convolutions to preserve time-frequency structure
   - Use dilated convolutions for larger temporal context
   - Add residual connections for deeper networks
   - Increase to 4-6 convolutional layers

2. **Temporal Modeling**
   - Replace fixed window with Bi-LSTM or Transformer layers
   - Use positional encodings for frame positions
   - Allow variable-length input sequences

3. **Training Enhancements**
   - **Data augmentation:**
     - Pitch shifts (±2 semitones)
     - Time stretches (0.9×-1.1×)
     - Gaussian noise injection
     - Mixup between examples
   - **Multiple SoundFonts:** Train on diverse timbres
   - **Diverse MIDI:** Include classical, jazz, pop, etc.
   - **Tuning variations:** A=435-445 Hz

4. **Inference Improvements**
   - Support CPU inference (remove CUDA hardcoding)
   - Add temporal smoothing (e.g., median filter over predictions)
   - Tune threshold per pitch or use learned thresholds

**Pros:** Builds on existing infrastructure, moderate effort
**Cons:** Still relies on hand-crafted VQT features

### Option 2: CREPE-Inspired End-to-End Model (Revolutionary)

**Approach:**
- Train a deep CNN on raw audio waveforms
- Multi-label classification for polyphonic pitch (unlike CREPE's monophonic)
- Use modern architectures (ResNets, EfficientNets)

**Architecture:**
```
Input: Raw audio (e.g., 1024 samples @ 22kHz)
  ↓
6-8 Conv1D layers with increasing channels (32→512)
  ↓
Global Average Pooling
  ↓
Dense layer(s)
  ↓
128 sigmoid outputs (multi-pitch)
```

**Training:**
- Same MIDI synthesis pipeline
- Augmentation: pitch shift, time stretch, noise, reverb
- Loss: BCE or Focal Loss (handles class imbalance)

**Pros:**
- End-to-end feature learning
- No dependency on VQT implementation
- Potentially better generalization

**Cons:**
- Requires more data and training time
- Larger model size
- May need GPU for real-time inference

### Option 3: Hybrid VQT + Transformer Model (Modern)

**Architecture:**
1. **Feature Extraction:** Use existing VQT (7×36 bins)
2. **2D Convolution:** Capture local time-frequency patterns
3. **Flatten/Reshape:** Convert to sequence of frame embeddings
4. **Transformer Encoder:** Model temporal dependencies with self-attention
5. **Per-Frame Classifier:** Predict 128 pitches for each frame

**Key Components:**
- Positional encoding for temporal positions
- Multi-head self-attention (e.g., 4-8 heads)
- Feed-forward networks with residual connections
- Layer normalization

**Training:**
- Sequence lengths: 10-50 frames
- Teacher forcing for autoregressive variants
- Attention visualization for interpretability

**Pros:**
- Models long-range musical dependencies (chord progressions, melodies)
- Attention maps provide interpretability
- State-of-the-art in many sequence tasks

**Cons:**
- Higher computational cost (attention is O(n²))
- Requires careful tuning and more data

### Option 4: Self-Supervised Pre-Training + Fine-Tuning

**Inspired by:** Toward Fully Self-Supervised Multi-Pitch Estimation (2024)

**Phase 1: Pre-Training (Self-Supervised)**
- Use contrastive learning on unlabeled audio
- Train model to predict masked VQT frames (like BERT)
- Or use audio-to-audio reconstruction tasks

**Phase 2: Fine-Tuning (Supervised)**
- Fine-tune pre-trained model on MIDI-synthesized data
- Requires much less labeled data
- Better generalization to real-world audio

**Pros:**
- Leverages unlimited unlabeled music
- Better generalization
- State-of-the-art approach in NLP/vision

**Cons:**
- Complex two-stage training
- Requires large unlabeled dataset

### Option 5: Hybrid Classical + Neural Approach

**Idea:** Use neural network to refine classical pitch detection

1. **Classical Baseline:** Use existing peak detection in VQT
2. **Neural Refinement:** Train a small network to:
   - Filter false positives
   - Add missing pitches
   - Adjust confidence scores

**Architecture:**
- Input: VQT + peak detection results
- Lightweight CNN or MLP
- Output: Refined pitch probabilities

**Pros:**
- Combines strengths of both approaches
- Smaller model, less training data needed
- Interpretable (classical method as baseline)

**Cons:**
- Upper-bounded by classical method performance

---

## Recommendations

### Short-Term (Low Effort, High Impact)

1. **Fix CUDA Hardcoding**
   - Change `Device::Cuda(0)` to `Device::cuda_if_available()`
   - Enable CPU fallback for broader usability

2. **Add Temporal Smoothing**
   - Apply median filter (3-5 frames) to predictions
   - Reduces flickering and false positives

3. **Evaluate on Real Audio**
   - Test on real recordings vs. synthesized data
   - Identify failure modes and generalization gaps

4. **Basic Data Augmentation**
   - Add pitch shifting (±1 semitone)
   - Add Gaussian noise
   - Retrain and compare performance

### Mid-Term (Moderate Effort)

5. **Enhanced Architecture**
   - Add 2D convolutions to preserve structure
   - Add 1-2 Bi-LSTM layers for temporal context
   - Increase model capacity to ~2-5M parameters

6. **Diverse Training Data**
   - Include multiple SoundFonts
   - Add MIDI from diverse genres (jazz, pop, rock)
   - Generate tuning variations

7. **Improved Training**
   - Add validation set and early stopping
   - Implement learning rate scheduling
   - Use Focal Loss to handle class imbalance

8. **Better Evaluation**
   - Per-pitch precision/recall curves
   - Compare against classical peak detection baseline
   - Evaluate on real instrument recordings

### Long-Term (High Effort, Research-Level)

9. **Transformer-Based Model**
   - Implement VQT + Transformer architecture
   - Explore self-attention for harmonic modeling
   - Benchmark against SOTA polyphonic pitch methods

10. **End-to-End Raw Audio Model**
    - Implement CREPE-inspired multi-pitch detector
    - Train on large diverse dataset
    - Optimize for real-time inference

11. **Self-Supervised Pre-Training**
    - Pre-train on unlabeled music corpora
    - Fine-tune on synthesized MIDI
    - Evaluate on real-world recordings

12. **Hybrid Ensemble**
    - Combine classical VQT peak detection + neural refinement
    - Use attention to focus on ambiguous regions
    - Leverage strengths of both paradigms

---

## Conclusion

The current ML feature represents an early, exploratory attempt at neural pitch detection in PitchVis. While the basic infrastructure is in place, the model suffers from:
- **Limited architecture** (simple CNN+MLP)
- **Narrow training data** (one SoundFont, mostly Bach)
- **Lack of temporal modeling** (fixed 5-frame window)
- **Practical constraints** (CUDA-only, no evaluation on real audio)

**The field has advanced significantly** since this implementation:
- Modern architectures (Transformers, dilated CNNs, attention mechanisms)
- End-to-end learning from raw audio (CREPE, SwiftF0)
- Self-supervised pre-training for better generalization
- Data augmentation and multi-stage refinement

**Recommended Path Forward:**

1. **Quick wins** (1-2 weeks): Fix CUDA, add smoothing, evaluate on real audio
2. **Solid improvement** (1-2 months): Enhanced architecture (2D CNN + LSTM), diverse data, better training
3. **SOTA performance** (3-6 months): Transformer-based or end-to-end raw audio model with self-supervised pre-training

The most pragmatic approach would be **Option 1 (Enhanced VQT-Based Model)** or **Option 3 (Hybrid VQT + Transformer)**, which build on existing infrastructure while incorporating modern techniques. For a research-oriented project, **Option 4 (Self-Supervised Pre-Training)** could achieve state-of-the-art performance on real-world audio.

---

## References

### Literature

1. [Polyphonic pitch tracking with deep layered learning](https://asa.scitation.org/doi/10.1121/10.0001468) (2018)
2. [Densely-connected Convolutional Recurrent Network for Fundamental Frequency Estimation](https://pmc.ncbi.nlm.nih.gov/articles/PMC12045132/) (DC-CRN, 2024)
3. [CREPE: A Convolutional Representation for Pitch Estimation](https://www.researchgate.net/publication/323276357_CREPE_A_Convolutional_Representation_for_Pitch_Estimation) (2018)
4. [Toward Fully Self-Supervised Multi-Pitch Estimation](https://arxiv.org/html/2402.15569v1) (2024)
5. [A Two-Stage Audio-Visual Fusion Piano Transcription Model Based on the Attention Mechanism](https://dl.acm.org/doi/10.1109/TASLP.2024.3426303) (2024)
6. [DeepF0: End-To-End Fundamental Frequency Estimation](https://www.semanticscholar.org/paper/DeepF0:-End-To-End-Fundamental-Frequency-Estimation-Singh-Wang/ab5751a31a08a6fe7631b7d54b5d6b670ae3a436)
7. [Convolutional neural network for robust pitch determination](https://www.researchgate.net/publication/288567213_Convolutional_neural_network_for_robust_pitch_determination)
8. [Deep Learning based Pitch Detection (CNN, LSTM)](https://praveenkrishna.medium.com/deep-learning-based-pitch-detection-cnn-lstm-3a2c5477c4e6)
9. [Creating musical features using multi-faceted, multi-task encoders based on transformers](https://www.nature.com/articles/s41598-023-36714-z) (2023)
10. [Traditional Machine Learning for Pitch Detection](https://arxiv.org/pdf/1903.01290) (2019)

### Code References

- Training pipeline: `pitchvis_train/src/train.rs`
- Model training: `pitchvis_train/train.py`
- Inference system: `pitchvis_viewer/src/ml_system.rs`
- Integration: `pitchvis_viewer/src/app/desktop_app.rs`
- VQT implementation: `pitchvis_analysis/src/vqt.rs`
- Display integration: `pitchvis_viewer/src/display_system/update.rs:308`

---

**Document Version:** 1.0
**Date:** 2025-11-22
**Author:** Claude (AI Assistant)
**Status:** Investigation Complete
