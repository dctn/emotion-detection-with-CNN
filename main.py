import streamlit as st
import torch
from torchvision import transforms
from torchvision.models import alexnet
import torch.nn as nn
import os
import requests
from PIL import Image

# ---------------- WATERMARK ----------------
st.markdown("""
<style>
.footer-watermark {
    position: fixed;
    bottom: 8px;
    right: 12px;
    opacity: 0.35;
    font-size: 14px;
}
.footer-watermark a {
    color: gray;
    text-decoration: none;
}
.footer-watermark a:hover {
    text-decoration: underline;
}
</style>

<div class="footer-watermark">
<a href="https://www.linkedin.com/in/hmaheswara/" target="_blank">
© 2025 Maheswara
</a>
</div>
""", unsafe_allow_html=True)

# CONFIG
st.title("Emotion Detection")

MODEL_URL = "https://critic.blr1.digitaloceanspaces.com/e_commerce/product_images/alex_model_v6_full_optim.pt"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "alex_model_v6_full_optim.pt")

CLASS_NAMES = [
    "angry", "disgust", "fear",
    "happy", "neutral", "sad", "surprise"
]

# MODEL LOADER
@st.cache_resource
def load_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL, stream=True, timeout=120)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model = alexnet(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 7)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model

# IMAGE UPLOAD
uploaded_image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

predict_btn = st.button("Predict")

# PREDICTION
if predict_btn:
    if uploaded_image is None:
        st.warning("Please upload an image")
        st.stop()

    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img_tensor = transform(image).unsqueeze(0)

    model = load_model()

    with torch.inference_mode():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()

    st.success(
        f"Predicted Emotion: **{CLASS_NAMES[pred_idx].capitalize()}**"
    )
