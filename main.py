import streamlit as st
import torch
from torchvision import transforms
from torchvision.models import alexnet
import torch.nn as nn
import os
import requests
from PIL import Image

st.markdown("""
<style>
.footer-watermark {
    position: fixed;
    bottom: 8px;
    right: 12px;
    opacity: 0.35;
    font-size: 35px;
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



st.title("Emotion Detection")

predict_img = st.file_uploader("Upload Image")
predict_btn = st.button("Predict")

if predict_btn and predict_img:
    st.success("Image Uploaded")

    MODEL_URL = "https://critic.blr1.digitaloceanspaces.com/e_commerce/product_images/alex_model_v6_full_optim.pt"
    MODEL_PATH = "alex_model_v6_full_optim.pt"


    @st.cache_resource
    def load_checkpoint():
        if not os.path.exists(MODEL_PATH):
            os.makedirs("models", exist_ok=True)
            with st.spinner("Downloading model..."):
                r = requests.get(MODEL_URL, stream=True, timeout=120)
                r.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        return torch.load(MODEL_PATH,weights_only=False)


    # img pre-process
    predict_img = Image.open(predict_img).convert('RGB')
    st.image(predict_img)

    test_img_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    predict_transformed_img = test_img_transformer(predict_img)

    class_name = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    ## prediction with model
    checkpoint = load_checkpoint()
    model = alexnet(pretrained=False)
    model.classifier[6] = nn.Linear(4096,7)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    with torch.inference_mode():
        pred_logits = model(predict_transformed_img.unsqueeze(0))
        pred_prob = torch.softmax(pred_logits, dim=1)
        predict = pred_prob.argmax(dim=1).item()

        st.title(f"your image is predicted as :red[{class_name[predict]}]".capitalize())

else:
    st.warning("Please upload an image")
