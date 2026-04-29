import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

# ─── GRAD-CAM CLASS ───────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach().cpu()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach().cpu()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().numpy()

        # Normalize
        if cam.max() != cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        cam = cv2.resize(cam, (224, 224))
        return cam

# ─── MAIN VISUALIZATION ───────────────────────────────────────
def visualize_gradcam(image_path, model_path):
    # Load model
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 14)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Setup Grad-CAM on last conv layer
    gradcam = GradCAM(model, model.layer4[-1].conv3)

    # Load image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized)

    input_tensor = transform(image).unsqueeze(0)
    input_tensor.requires_grad_(True)

    # Get predictions
    with torch.no_grad():
        output_probs = torch.sigmoid(model(input_tensor))

    predicted_class = output_probs.argmax().item()
    confidence = output_probs[0, predicted_class].item()

    # Generate Grad-CAM
    cam = gradcam.generate(input_tensor, predicted_class)

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = (0.5 * img_array + 0.5 * heatmap_rgb).astype(np.uint8)

    # Plot all 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_array, cmap='gray')
    axes[0].set_title('Original X-ray', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap\n(Red = High Attention)', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay\nPredicted: {CLASSES[predicted_class]}\nConfidence: {confidence:.2%}',
                      fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle('ChestAI — Medical X-ray Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('gradcam_result.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nPredicted disease: {CLASSES[predicted_class]}")
    print(f"Confidence: {confidence:.2%}")
    print("Saved: gradcam_result.png ✅")

# ─── RUN ──────────────────────────────────────────────────────
visualize_gradcam(
    r'C:\Users\shaik\OneDrive\Desktop\medical-ai-portfolio\images\00000001_000.png',
    'chest_xray_model.pth'
)