import os
from PIL import Image
import torch
import streamlit as st
from model_builder import SignatureCNN
from prediction import predict_signature

# 1. Page config
st.set_page_config(
    page_title="✨ Signature Verification",
)

# 2. CSS injection for matte-black theme
st.markdown(
    """
    <style>
      [data-testid="stAppViewContainer"] { background-color: #000000; }
      [data-testid="stSidebar"] > div:first-child { background-color: #1c1c1c; }
      
      .css-1d391kg, .css-1d391kg * { color: #FFF !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# 3. Constants
HERE            = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(HERE, "best_signature_model.pth")
IMG_SIZE        = (224, 224)
CLASS_NAMES     = ["forged_images", "real_images"]

# 4. Model loader
@st.cache_resource
def load_model(device):
    model = SignatureCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device).eval()
    return model

# 5. Main app
def main():
    st.title("✨ Signature Verification")
    st.write("Upload a signature image below →")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(device)

    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])
    if not uploaded_file:
        return

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Signature", use_column_width=True)

    with st.spinner("Predicting…"):
        tmp_path   = os.path.join(HERE, "tmp_signature.png")
        img.save(tmp_path)
        prediction = predict_signature(
            model, tmp_path, device,
            img_size=IMG_SIZE, class_names=CLASS_NAMES
        )

    st.success(f"Prediction: **{prediction}**")

if __name__ == "__main__":
    main()
