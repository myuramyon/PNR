import cv2
import torch
import easyocr
import pandas as pd

def detect_and_read(video_path):
    # Initialize OCR reader
    reader = easyocr.Reader(['en'], gpu=False)

    # Load YOLO model (switch to YOLOv8 via Ultralytics if needed)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame)
        detections = results.pandas().xyxy[0]

        # Optional: print detections
        print(f"\nFrame {frame_num} Detections:")
        print(detections[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']])

        # Extract plate region (if 'license-plate' class is used, replace logic here)
        for _, row in detections.iterrows():
            xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            crop = frame[ymin:ymax, xmin:xmax]
            ocr_result = reader.readtext(crop)
            print("Detected text:", [text[1] for text in ocr_result])

        frame_num += 1

    cap.release()
