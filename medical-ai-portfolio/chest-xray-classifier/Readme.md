#  ChestAI — Chest X-ray Disease Classifier

**Medical AI Project | Shaik Muskan | Vardhaman College of Engineering**

## Overview
Deep learning model detecting 14 chest diseases from X-rays using 
ResNet50 transfer learning with Grad-CAM explainability.

## Dataset
- NIH Chest X-ray Dataset: 112,120 images, 14 disease classes
- Source: National Institutes of Health

## Model Architecture
- Base: ResNet50 (pretrained on ImageNet)
- Transfer learning with custom classification head
- Multi-label classification (BCEWithLogitsLoss)
- Grad-CAM heatmaps for explainability

## Results
| Metric | Score |
|--------|-------|
| Training Loss | ~0.18 |
| Validation AUC | ~0.82 |

## Disease Classes Detected
Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule,
Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema,
Fibrosis, Pleural Thickening, Hernia

## Tech Stack
PyTorch · ResNet50 · Grad-CAM · Streamlit · Pandas · NumPy

## Run Locally
pip install -r requirements.txt
streamlit run app.py

## Why This Matters
This type of AI-assisted radiology is used by companies like 
Qure.ai and Siemens Healthineers to screen millions of patients.