import cv2
from utils.openvino_utils import OpenVINOYOLODetector
import json

with open("yolo-config.json", "r") as f:
    config = json.load(f)

model_path = config.get("model_path", "yolo12n.pt")
conf_thesh = config.get("conf_thesh", 0.5)
iou_thesh = config.get("iou_thesh", 0.5)
max_det = config.get("max_det", 80)
imgsz = config.get("imgsz", 640)
verbose = config.get("verbose", True)

model = OpenVINOYOLODetector(model_path)

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

img = cv2.imread("image.jpg")
model.predict(img, conf=conf_thesh, iou=iou_thesh, max_det=max_det, verbose=verbose)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(frame, conf=conf_thesh, iou=iou_thesh, max_det=max_det, verbose=verbose)
    
    r = results[0].numpy()

    boxes = r.boxes
    names = r.names

    if boxes is not None:

        xyxy = boxes.xyxy
        conf = boxes.conf
        cls  = boxes.cls

        for box, score, c in zip(xyxy, conf, cls):

            x1, y1, x2, y2 = map(int, box)
            try:
                label = names[int(c)]
            except Exception as e:
                label = str(f"Unknown {c}:")

            text = f"{label} {score:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

    cv2.imshow("OpenVINO YOLO", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()