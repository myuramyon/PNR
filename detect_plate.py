
def detect_license_plate(frame, model):
    results = model(frame)
    plates = []
    for *box, conf, cls in results.xyxy[0]:
        plates.append(frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])])
    return plates
