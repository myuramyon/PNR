
import streamlit as st
import tempfile
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import easyocr
import os
from utils import extract_frames, preprocess_plate
from detect_plate import detect_license_plate

reader = easyocr.Reader(['en'])

@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt', force_reload=True)

model = load_model()

def read_plate_text(img):
    try:
        text = ""
        for _, detected_text, conf in reader.readtext(img):
            if conf > 0.7:
                text += detected_text + " "
        return text.strip()
    except Exception as e:
        st.error(f"OCR Error: {e}")
        return ""

st.set_page_config(page_title="🇮🇳 Indian Number Plate Recognition", layout="centered")
st.title("🇮🇳 Automatic Number Plate Recognition (ANPR)")
st.caption("Upload a video or image to detect Indian license plates using YOLOv5 + EasyOCR.")

video = st.file_uploader("📽️ Upload Video", type=["mp4", "avi", "mov"])
image = st.file_uploader("📸 Or Upload Image", type=["jpg", "jpeg", "png"])

results = []

if video:
    st.info("🔍 Processing video...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp_vid:
        tmp_vid.write(video.read())
        frames = extract_frames(tmp_vid.name, interval=15)
        for i, frame in enumerate(frames):
            for plate in detect_license_plate(frame, model):
                plate_img = preprocess_plate(plate)
                text = read_plate_text(plate_img)
                if text:
                    results.append({"Frame": i, "Plate": text})

if image:
    st.info("🔍 Processing image...")
    img_arr = np.asarray(bytearray(image.read()), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, 1)
    for plate in detect_license_plate(frame, model):
        plate_img = preprocess_plate(plate)
        text = read_plate_text(plate_img)
        if text:
            results.append({"Plate": text})

if results:
    df = pd.DataFrame(results)
    st.success("✅ Detection Complete!")
    st.dataframe(df)
    st.download_button("📥 Download CSV", df.to_csv(index=False).encode(), "plates.csv", "text/csv")
else:
    st.warning("⚠️ No license plates detected yet.")
