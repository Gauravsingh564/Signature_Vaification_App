import streamlit as st
import sys
import os
import torch
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT_PATH = os.path.join(HERE, "best_signature_model.pth")


from model_builder import SignatureCNN
from prediction import predict_signature

# Paths and settings
IMG_SIZE = (224, 224)
CLASS_NAMES = ["forged_images", "real_images"]

@st.cache_resource
def load_model(device):
    model = SignatureCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
   st.markdown(
    '''
    <style>
      /* Full-page gradient background */
      .stApp {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, #1a2a6c 0%, #b21f1f 50%, #fdbb2d 100%);
      }
      /* Shooting star style */
      .shooting-star {
        position: absolute;
        top: -10px;
        left: -10px;
        width: 3px;
        height: 100px;
        background: linear-gradient(-45deg, white, rgba(255,255,255,0));
        opacity: 0;
        transform: rotate(45deg);
        animation: shoot 1s ease-out infinite;
      }
      @keyframes shoot {
        0% {
          opacity: 0;
          transform: translate(-100px, 0) rotate(45deg);
        }
        10% {
          opacity: 1;
        }
        100% {
          opacity: 0;
          transform: translate(800px, 600px) rotate(45deg);
        }
      }
    </style>
    <div class="shooting-star"></div>
    ''',
    unsafe_allow_html=True
)
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
