# Hybrid CNN-LSTM Neural Network Architecture - Spatial-Temporal Modeling

<p align="center">
  <a href="https://sohail.devsil.com"><strong>Portfolio</strong></a> | 
  <a href="https://github.com/sohailahmed-bsrh"><strong>GitHub Profile</strong></a> | 
  <a href="https://github.com/sohailahmed-bsrh/hybrid-cnn-lstm-architecture"><strong>Source Repository</strong></a>
</p>

---

## Executive Summary & Problem Statement

Standard deep learning architectures excel at isolated domains: Convolutional Neural Networks (CNNs) extract hierarchical spatial features from static matrices, while Recurrent Neural Networks (such as LSTMs) model temporal sequence dependencies. However, complex multi-modal problems (such as video action classification, speech sentiment analysis, and continuous biometric signal tracking) require modeling spatial patterns that evolve over time.

This project implements a unified **Hybrid CNN-LSTM Deep Learning Architecture**. By coupling a 2D/1D Convolutional feature extraction backbone with a Long Short-Term Memory temporal recurrence layer, the network simultaneously captures spatial feature representations and sequential temporal dynamics.

<div align="center">
  <img src="https://raw.githubusercontent.com/sohailahmed-bsrh/myportfolio/main/assets/images/thumbnail/cnn-vision-lstm-nlp.jpeg" width="650" alt="Hybrid CNN-LSTM Architecture"/>
  <p><em>Unified deep learning pipeline coupling convolutional spatial representations with recurrent temporal memory.</em></p>
</div>

---

## Real-World Industry Applications

- **Video Action & Activity Recognition**: Analyzes sequential video frames to classify human activities and gestures in real time.
- **Biomedical Signal Processing**: Classifies multi-lead ECG and EEG waveforms where spatial wave morphology changes across temporal cycles.
- **Sequential Sentiment Analysis & NLP**: Combines n-gram spatial pattern extraction via 1D convolutions with long-range semantic context tracking via LSTM cells.
- **Industrial Sensor Anomaly Tracking**: Monitors multi-sensor telemetry across industrial manufacturing pipelines.

---

## Technical Architecture & Engineering Walkthrough

```
                      [Input Tensor: Sequence of Spatial Frames]
                                         │
                                         ▼
            [TimeDistributed Convolutional Layers (Conv2D / Conv1D)]
                        (Spatial Feature Extraction)
                                         │
                                         ▼
            [Spatial Pooling & Feature Flattening (MaxPooling / GAP)]
                                         │
                                         ▼
                    [Spatial Feature Vectors across Time]
                                         │
                                         ▼
                 [Stacked LSTM Recurrent Layers (Memory Cells)]
                        (Temporal Dynamics & State Retention)
                                         │
                                         ▼
             [Fully Connected Dense Layers with Dropout Regularization]
                                         │
                                         ▼
               [Softmax / Sigmoid Classification & Risk Scoring]
```

### Step 1: Multi-Modal Data Preprocessing & Formatting
Raw inputs are structured into sequential multi-dimensional tensors `(batch_size, sequence_length, height, width, channels)` for visual inputs or `(batch_size, time_steps, features)` for sensor signals.

### Step 2: TimeDistributed Convolutional Feature Extraction
Wraps convolutional layers within a `TimeDistributed` wrapper in Keras/TensorFlow. This enforces consistent spatial weight kernels across every consecutive time step without memory leakage.

### Step 3: Recurrent Sequence Modeling via LSTM
Spatial feature representations output by the convolutional stage are routed directly into stacked LSTM cells. The input gate, forget gate, and cell state preserve long-range dependencies across long temporal sequences.

### Step 4: Regularization & Optimization
Employs Spatial Dropout, Recurrent Dropout, and Batch Normalization to prevent co-adaptation of feature detectors. Uses the Adam optimizer with scheduled learning rate decay.

### Step 5: Empirical Evaluation & Ablation Study
Compares performance metrics (Accuracy, F1-Score, ROC-AUC, Confusion Matrix) against standalone CNN and standalone LSTM baselines to confirm the hybrid architecture's superiority.

---

## Technology Stack & Implementation Specifications

| Component | Technology | Version | Architectural Function |
| :--- | :--- | :--- | :--- |
| **Deep Learning Framework** | TensorFlow / Keras | 2.15+ | Modular CNN-LSTM layer construction and training |
| **Computation Engine** | NumPy | Latest | Multi-dimensional tensor transformation and normalization |
| **Data Handling** | Pandas | 2.x | Sequence metadata indexing and split management |
| **Metric Evaluation** | Scikit-Learn | 1.3+ | Multi-class classification metrics and ROC curve generation |
| **Visualization** | Matplotlib / Seaborn | 3.8+ | Training loss/accuracy curves and confusion matrices |

---

## Installation & Execution Guide

```bash
git clone https://github.com/sohailahmed-bsrh/hybrid-cnn-lstm-architecture.git
cd hybrid-cnn-lstm-architecture
python -m venv venv
# Windows: venv\Scriptsctivate | Unix: source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

---

## Author & Contact Information

<table width="100%">
  <tr>
    <td width="150" align="center" valign="middle">
      <img src="https://avatars.githubusercontent.com/u/231165170?v=4" width="120" style="border-radius: 50%;" alt="Sohail Ahmed"/>
    </td>
    <td>
      <h3>Sohail Ahmed</h3>
      <p><strong>Senior Software Engineer & AI Systems Architect</strong></p>
      <p>Specializing in Full-Stack AI Application Development, Computer Vision, Deep Learning Pipelines, and Scalable Automation Architectures.</p>
      <p>
        <a href="https://sohail.devsil.com"><strong>Portfolio Website</strong></a> | 
        <a href="https://github.com/sohailahmed-bsrh"><strong>GitHub Profile</strong></a> | 
        <a href="https://www.linkedin.com/in/sohail-ahmed-546314316/"><strong>LinkedIn</strong></a> | 
        <a href="mailto:se906403@gmail.com"><strong>Email Contact</strong></a>
      </p>
    </td>
  </tr>
</table>
