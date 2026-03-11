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

    if boxes is not None and len(boxes) > 0:

        xyxy = boxes.xyxy
        conf = boxes.conf
        cls  = boxes.cls

        for box, score, c in zip(xyxy, conf, cls):

            x1, y1, x2, y2 = map(int, box)

            try:
                label = names[int(c)]
            except Exception:
                label = f"Unknown {c}"

            text = f"{label} {score:.2f}"

            # -----------------------------
            # Draw bounding box (GREEN)
            # -----------------------------
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0,255,0),
                2
            )

            # -----------------------------
            # Draw centroid (BLUE DOT)
            # -----------------------------
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            cv2.circle(
                frame,
                (cx, cy),
                4,
                (255,0,0),
                -1
            )

            # -----------------------------
            # Text background rectangle
            # -----------------------------
            (w, h), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            cv2.rectangle(
                frame,
                (x1, y1 - h - 10),
                (x1 + w, y1),
                (0,255,0),
                -1
            )

            # -----------------------------
            # Draw label text (WHITE)
            # -----------------------------
            cv2.putText(
                frame,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2
            )

    cv2.imshow("OpenVINO YOLO", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()