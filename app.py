import streamlit as st
import torch
import tempfile
from detect_plate import detect_plate

@st.cache_resource
def load_model():
    model = torch.load('models/best.pt', map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()

st.title("🚗 ANPR: Indian License Plate Recognition")
uploaded_file = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    result_img, text = detect_plate(model, tmp_path)
    st.image(result_img, caption="Detected Plate", use_column_width=True)
    st.success(f"Recognized Text: {text}")
