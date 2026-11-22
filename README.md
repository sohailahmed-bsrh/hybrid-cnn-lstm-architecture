# Hybrid Multi-Modal CNN-LSTM Deep Neural Network

> Dual-stream deep architecture fusing spatial vision features with recurrent sequential text models.

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github)](https://github.com/sohailahmed-bsrh/hybrid-cnn-lstm-architecture)
[![Live Showcase](https://img.shields.io/badge/Live-Showcase-0A66C2?style=flat-square&logo=googlechrome)](https://sohail.devsil.com/projects.html)
[![Author](https://img.shields.io/badge/Author-Sohail%20Ahmed-20BEFF?style=flat-square)](https://sohail.devsil.com)

---

## Overview

Multi-modal tasks require disparate deep learning architectures capable of processing both 2D spatial features and temporal sequences.

This project provides an end-to-end, reproducible implementation engineered to solve this challenge through verified data transformations, reliable algorithms, and production-ready architecture.

---

## Key Features

- Convolutional Neural Network feature extraction stream for spatial pattern classification
- Long Short-Term Memory (LSTM) recurrent network processing sequential semantic data
- End-to-end model checkpointing, loss curve evaluation, and validation metrics
- Modular Jupyter notebook walkthrough with comparative performance analysis

---

## System Architecture

```mermaid
flowchart TD
    subgraph Vision Stream
        ImageInput[Spatial Image Input] --> ConvLayers[Conv2D + MaxPool Layers]
        ConvLayers --> SpatialEmbed[Spatial Feature Vector]
    end
    subgraph NLP Stream
        TextInput[Text Sequence Input] --> EmbeddingLayer[Word Embedding Layer]
        EmbeddingLayer --> LSTMLayers[LSTM Recurrent Unit]
        LSTMLayers --> SequenceEmbed[Temporal Feature Vector]
    end
    SpatialEmbed --> Evaluation[Evaluation Metrics & Loss Validation]
    SequenceEmbed --> Evaluation
```

---

## Technology Stack

| Technology | Version / Spec | Role in Architecture |
| :--- | :--- | :--- |
| **TensorFlow / Keras** | `2.12+` | Convolutional and recurrent neural network implementation |
| **OpenCV** | `4.8+` | Image transformation and visual normalization |
| **NumPy & Pandas** | `2.0+` | Tensor manipulations and dataset structuring |

---

## Repository Structure

```text
hybrid-cnn-lstm-architecture/
|-- notebooks/
|   `-- Deep_Learning_Project_using_CNN_(Computer_Vision)_and_LSTM_(NLP).ipynb
|-- data/
|-- .gitignore
|-- requirements.txt
`-- README.md
```

---

## Installation and Execution

### Prerequisites
- Python 3.10 or higher
- Git version control
- Virtual environment manager (`venv`)

### Setup Instructions
1. Clone the repository:
```bash
git clone https://github.com/sohailahmed-bsrh/hybrid-cnn-lstm-architecture.git
cd hybrid-cnn-lstm-architecture
```

2. Configure environment and dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

---

## Interface and Verification
The system produces verifiable execution logs, prediction matrices, or interactive browser interfaces depending on the target mode. Verify operation by executing the commands specified above or inspecting the demonstration notebooks.

---

## Author and Contact
Developed by **Sohail Ahmed** — Full Stack AI Engineer & Machine Learning Specialist.
- **Portfolio:** [sohail.devsil.com](https://sohail.devsil.com)
- **GitHub:** [@sohailahmed-bsrh](https://github.com/sohailahmed-bsrh)
- **Email:** [sohail.bsrh@gmail.com](mailto:sohail.bsrh@gmail.com)

---

## Author & Contact Information

<table width="100%" align="center">
  <tr>
    <td align="center" style="padding: 20px;">
      <h3>Sohail Ahmed</h3>
      <p><strong>Full-Stack Artificial Intelligence & Computer Vision Engineer</strong></p>
      <p>
        <a href="https://sohail.devsil.com"><img src="https://img.shields.io/badge/Live_Portfolio-sohail.devsil.com-0052FF?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Portfolio"/></a>
        <a href="https://github.com/sohailahmed-bsrh"><img src="https://img.shields.io/badge/GitHub-Profile-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/></a>
        <a href="https://linkedin.com/in/sohail-ahmed-dev"><img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/></a>
        <a href="mailto:sohail.bsrh@gmail.com"><img src="https://img.shields.io/badge/Email-Contact_Me-EA4335?style=for-the-badge&logo=gmail&logoColor=white" alt="Email"/></a>
      </p>
      <p><em>Available for technical roles in Artificial Intelligence, Computer Vision, and Full-Stack Engineering.</em></p>
      <hr style="border: 0; height: 1px; background: #30363d; margin: 15px 0;">
      <small>Distributed under the MIT License.</small>
    </td>
  </tr>
</table>
