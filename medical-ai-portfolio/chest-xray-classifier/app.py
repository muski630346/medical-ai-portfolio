import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

st.set_page_config(page_title="ChestAI — Medical X-ray Analyzer", layout="wide")
st.title("🫁 ChestAI — Chest X-ray Disease Classifier")
st.markdown("**Medical AI Project by Shaik Muskan | Vardhaman College of Engineering**")
st.markdown("*Detects 14 diseases from chest X-rays using ResNet50 + Grad-CAM explainability*")

@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 14)
    model.load_state_dict(torch.load('chest_xray_model.pth', map_location='cpu'))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert('RGB')

    with col1:
        st.subheader("Uploaded X-ray")
        st.image(image, use_column_width=True)

    model = load_model()
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor))

    predictions = output[0].numpy()

    with col2:
        st.subheader("Disease Detection Results")
        for i, (cls, prob) in enumerate(zip(CLASSES, predictions)):
            if prob > 0.3:
                st.error(f"⚠️ {cls}: {prob:.1%}")
            elif prob > 0.15:
                st.warning(f"⚡ {cls}: {prob:.1%}")
            else:
                st.success(f"✓ {cls}: {prob:.1%}")

    st.subheader("Top Predictions")
    top_indices = predictions.argsort()[-5:][::-1]
    fig, ax = plt.subplots(figsize=(10, 4))
    top_classes = [CLASSES[i] for i in top_indices]
    top_probs = [predictions[i] for i in top_indices]
    bars = ax.barh(top_classes, top_probs, color=['#ef4444' if p > 0.3 else '#f59e0b' if p > 0.15 else '#22c55e' for p in top_probs])
    ax.set_xlabel('Probability')
    ax.set_title('Disease Probability Scores')
    ax.set_xlim(0, 1)
    st.pyplot(fig)

    st.info("⚕️ **Disclaimer:** This AI tool is for research purposes only. Always consult a qualified radiologist for medical diagnosis.")