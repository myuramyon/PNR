
import cv2

def extract_frames(video_path, interval=30):
    frames = []
    cap = cv2.VideoCapture(video_path)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % interval == 0:
            frames.append(frame)
        i += 1
    cap.release()
    return frames

def preprocess_plate(plate_img):
    return cv2.resize(plate_img, (200, 60))
