# detect_plate.py

import cv2
import torch
import easyocr
import pandas as pd

def detect_plate(video_path):
    reader = easyocr.Reader(['en'], gpu=False)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    cap = cv2.VideoCapture(video_path)
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        df = results.pandas().xyxy[0]

        print(f"\nFrame {frame_num} Detections:")
        print(df[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']])

        for _, row in df.iterrows():
            xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            plate_crop = frame[ymin:ymax, xmin:xmax]
            text = reader.readtext(plate_crop)
            print("Detected text:", [t[1] for t in text])

        frame_num += 1

    cap.release()
