import streamlit as st
import os
import torch
from PIL import Image
from model_builder import SignatureCNN
from prediction import predict_signature

# ─── 1. Script directory & checkpoint ─────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(HERE, "best_signature_model.pth")

# ─── 2. (Optional) gradient background only ────────────────────────────
# you can replace this block with your shooting-star CSS if you like
st.markdown(
    """
    <style>
      /* Full-page gradient background */
      .stApp {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, #1a2a6c 0%, #b21f1f 50%, #fdbb2d 100%);
      }

      /* Shooting star base style */
      .shooting-star {
        position: absolute;
        top: -20px;
        left: -20px;
        width: 2px;
        height: 120px;
        background: linear-gradient(-45deg, #fff, rgba(255,255,255,0));
        opacity: 0;
        transform: rotate(45deg);
        animation: shoot 1.5s ease-out infinite;
      }

      /* Keyframes for the shooting star */
      @keyframes shoot {
        0%   { opacity: 0; transform: translate(-100px, 0) rotate(45deg); }
        10%  { opacity: 1; }
        100% { opacity: 0; transform: translate(900px, 700px) rotate(45deg); }
      }
    </style>

    <!-- Insert as many of these as you like for multiple stars -->
    <div class="shooting-star" style="animation-delay: 0s;"></div>
    <div class="shooting-star" style="animation-delay: 0.7s;"></div>
    <div class="shooting-star" style="animation-delay: 1.4s;"></div>
    <div class="shooting-star" style="animation-delay: 2.1s;"></div>
    """,
    unsafe_allow_html=True
)


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
            model,
            tmp_path,
            device,
            img_size=IMG_SIZE,
            class_names=CLASS_NAMES
        )

    st.success(f"Prediction: **{prediction}**")

if __name__ == "__main__":
    main()
