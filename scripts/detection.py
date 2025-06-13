import cv2
import torch
import easyocr

def detect_and_read(video_path):
    reader = easyocr.Reader(['en'], gpu=False)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov8n.pt', force_reload=True)

    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        df = results.pandas().xyxy[0]
        print(f"Frame {frame_num}:")
        print(df[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']])
        frame_num += 1

    cap.release()
