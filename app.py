import streamlit as st
import os
import torch
from PIL import Image
from model_builder import SignatureCNN
from prediction import predict_signature

# ─── 1. Script directory & checkpoint ─────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(HERE, "best_signature_model.pth")

# ─── 2. Inject gradient + shooting-star CSS ───────────────────────────


# ─── 3. App constants ───────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["forged_images", "real_images"]

# ─── 4. Model loader ────────────────────────────────────────────────────
@st.cache_resource
def load_model(device):
    model = SignatureCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device).eval()
    return model

# ─── 5. Main app ───────────────────────────────────────────────────────
def main():
   st.markdown(
    """
    <style>
      /* Targets the main app container */
      .stApp {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        background-attachment: fixed;
      }
    </style>
    """,
    unsafe_allow_html=True
)
    st.title("✨ Signature Verification")
    st.write("Upload a signature image below →")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    # File upload
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])
    if not uploaded_file:
        return

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Signature", use_column_width=True)

    # Predict
    with st.spinner("Predicting…"):
        tmp_path = os.path.join(HERE, "tmp_signature.png")
        img.save(tmp_path)
        prediction = predict_signature(
            model, tmp_path, device,
            img_size=IMG_SIZE, class_names=CLASS_NAMES
        )

    st.success(f"Prediction: **{prediction}**")

if __name__ == "__main__":
    main()
