import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from src.model import build_model
import pandas as pd

# ── Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="FruitGuard",
    page_icon="🍎",
    layout="centered"
)

CLASSES = ['freshapples', 'freshbanana', 'freshoranges',
           'rottenapples', 'rottenbanana', 'rottenoranges']

LABELS = {
    'freshapples':  ('🍎 Fresh Apple',   'fresh'),
    'freshbanana':  ('🍌 Fresh Banana',  'fresh'),
    'freshoranges': ('🍊 Fresh Orange',  'fresh'),
    'rottenapples': ('🍎 Rotten Apple',  'rotten'),
    'rottenbanana': ('🍌 Rotten Banana', 'rotten'),
    'rottenoranges':('🍊 Rotten Orange', 'rotten'),
}

# ── Load model ───────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(ROOT / "best_model.pth", map_location=device))
    for param in model.parameters():
        param.requires_grad = True
    model.eval()
    model.to(device)
    return model, device

model, device = load_model()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Grad-CAM ─────────────────────────────────────────────
def run_gradcam(img_array, input_tensor, pred_idx):
    cam = GradCAMPlusPlus(model=model, target_layers=[model.features[-1]])
    with torch.enable_grad():
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=[ClassifierOutputTarget(pred_idx)])
    return show_cam_on_image(img_array.astype(np.float32),
                             grayscale_cam[0], use_rgb=True)

# ── Predict ───────────────────────────────────────────────
def predict(image: Image.Image):
    img_resized = image.resize((224, 224)).convert("RGB")
    img_array   = np.array(img_resized) / 255.0
    input_tensor = preprocess(img_resized).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    with torch.enable_grad():
        output = model(input_tensor)

    probs    = torch.softmax(output, dim=1)[0]
    pred_idx = probs.argmax().item()
    pred_class = CLASSES[pred_idx]
    confidence = probs[pred_idx].item()

    cam_image = run_gradcam(img_array, input_tensor, pred_idx)

    return pred_class, confidence, probs.detach().cpu().numpy(), cam_image

# ── UI ────────────────────────────────────────────────────
st.title("🍎 FruitGuard")
st.markdown("**AI-powered fruit quality inspection** — Upload a photo to detect freshness.")
st.divider()

uploaded = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    pred_class, confidence, probs, cam_image = predict(image)

    label_name, status = LABELS[pred_class]
    color = "green" if status == "fresh" else "red"

    st.markdown(f"### :{color}[{label_name}]")
    st.markdown(f"**Confidence:** {confidence:.1%}")
    st.progress(confidence)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        st.image(cam_image, caption="Grad-CAM — where the model looked",
                 use_container_width=True)

    st.divider()
    st.markdown("#### Confidence per class")
    df = pd.DataFrame({
        "Class": CLASSES,
        "Confidence": [f"{p:.1%}" for p in probs],
        "Score": probs
    }).sort_values("Score", ascending=False)
    st.dataframe(df[["Class", "Confidence"]], use_container_width=True, hide_index=True)

    # ── History ───────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "File": uploaded.name,
        "Prediction": label_name,
        "Status": status.upper(),
        "Confidence": f"{confidence:.1%}"
    })

    if len(st.session_state.history) > 1:
        st.divider()
        st.markdown("#### 📋 Prediction History")
        st.dataframe(pd.DataFrame(st.session_state.history),
                     use_container_width=True, hide_index=True)