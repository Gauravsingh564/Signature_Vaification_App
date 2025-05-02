import streamlit as st
import sys
import os
import torch
from PIL import Image
# ─── 1. Define where “app.py” lives ────────────────────────────────
HERE = os.path.dirname(__file__)

# ─── 2. (Optional) Debug listing ────────────────────────────────────
st.write("App folder:",os.getcwd())
st.write("Contents:", os.listdir(HERE))
# ─── 1. Define your checkpoint location ────────────────────────────────

CHECKPOINT_PATH = os.path.join(HERE, "best_signature_model(1).pth")

from model_builder import SignatureCNN
from prediction import predict_signature

# Paths and settings
IMG_SIZE = (224, 224)
CLASS_NAMES = ["fored_images", "real_images"]

@st.cache_resource
def load_model(device):
    model = SignatureCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    st.title("Signature Verification")
    st.write("Upload a signature image, and this app will predict whether it is forged or real.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.write("")

        with st.spinner("Predicting..."):
            tmp_path = "/tmp/uploaded_signature.png"
            img.save(tmp_path)
            prediction = predict_signature(
                model,
                tmp_path,
                device,
                img_size=IMG_SIZE,
                class_names=CLASS_NAMES
            )

        st.success(f"Prediction: **{prediction}**")

if __name__ == "__main__":
    main()
