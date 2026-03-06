import cv2
from utils.openvino_utils import OpenVINOYOLODetector

model = OpenVINOYOLODetector("yolo12n.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

img = cv2.imread("image.jpg")
model.predict(img, conf=0.25, iou=0.7, max_det=80, verbose=False)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(frame, conf=0.25, iou=0.7, max_det=80, verbose=False)
    r = results[0]

    boxes = r.boxes
    names = r.names

    if boxes is not None:

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls  = boxes.cls.cpu().numpy()

        for box, score, c in zip(xyxy, conf, cls):

            x1, y1, x2, y2 = map(int, box)

            label = names[int(c)]
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