# Brain MRI Vision-Language Model Pipeline

This repository contains a pipeline for automated radiological analysis of brain MRI studies using the **MedGemma** vision-language model. It processes DICOM series, extracts representative slices, and generates structured radiology reports or clinical routing decisions.

## 🚀 Overview

The pipeline leverages `google/medgemma-1.5-4b-it` to analyze multimodal brain imaging data. It is designed to handle multi-series studies and provides two primary workflows:
1.  **Detailed Radiology Reporting**: Comprehensive analysis across sequences (T1, T2, FLAIR, etc.) with Chain-of-Thought reasoning.
2.  **Clinical Routing**: Rapid classification of studies based on the presence of anomalies and tumors.

## 🛠️ Environment Setup

Before running the scripts, ensure you have the necessary environment variables configured for Hugging Face and PyTorch.

```bash
export HF_HOME=/data/huggingface
export TRANSFORMERS_CACHE=/data/huggingface/transformers
export HF_DATASETS_CACHE=/data/huggingface/datasets
export TORCH_HOME=/data/torch
export HF_TOKEN=""
```

### Installation

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

> [!NOTE]
> This project requires CUDA-enabled GPU support (cu118) for optimal performance.

## 📂 Project Structure

- `med_gemma_FE.py`: Feature Extraction and detailed radiology report generation script.
- `routing_anomaly.py`: Clinical routing script for anomaly and tumor detection.
- `requirements.txt`: Python package dependencies.
- `PATIENT_*/`: Directory structure containing DICOM studies for different patients.
- `reports/`: Directory structure containing radiology reports.

## 🖥️ Usage

### 1. Generating a Radiology Report
Run the following command to process a specific patient study and generate a detailed report:

```bash
python med_gemma_FE.py
```

### 2. Clinical Routing (Anomaly/Tumor Detection)
To perform rapid clinical routing on patient data:

```bash
python routing_anomaly.py
```

## 📖 Key Components

- **DICOM Processing**: Normalizes MRI slices to 8-bit RGB and samples representative slices for VLM input.
- **Multimodal Prompting**: Combines series metadata with visual data to provide context to the model.
- **Chain-of-Thought (CoT)**: Enforces step-by-step reasoning for more accurate and interpretable medical analysis.

---

