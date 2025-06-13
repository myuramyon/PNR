import cv2
import numpy as np
import easyocr
from PIL import Image

reader = easyocr.Reader(['en'])

def detect_plate(model, image_path):
    image = cv2.imread(image_path)
    results = model([image])
    boxes = results.xyxy[0]

    for *xyxy, conf, cls in boxes:
        x1, y1, x2, y2 = map(int, xyxy)
        plate_crop = image[y1:y2, x1:x2]
        text_result = reader.readtext(plate_crop, detail=0)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return pil_image, " ".join(text_result)
