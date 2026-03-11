import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# ================================
# Load model hand landmarker
# ================================

model_path = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=4,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.HandLandmarker.create_from_options(options)
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
    annotated_frame = frame.copy()

    if not ret:
        break
    
    h, w, _ = frame.shape

    # Convert ke RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert ke MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )
    
    # ================================
    # Inference
    # ================================

    timestamp_ms = int(time.time() * 1000)
    hand_result = detector.detect_for_video(mp_image, timestamp_ms)
    # ================================
    # Process hands
    # ================================
    HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
    ]
    if hand_result.hand_landmarks:

        for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):

            label = handedness[0].category_name
            score = handedness[0].score

            points = []

            for lm in hand_landmarks:
                px = int(lm.x * w)
                py = int(lm.y * h)
                points.append((px, py))

            # =========================
            # Draw skeleton (HIJAU)
            # =========================
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection

                x1, y1 = points[start_idx]
                x2, y2 = points[end_idx]

                cv2.line(annotated_frame,(x1,y1),(x2,y2),(0,255,0),2)

            # =========================
            # Draw keypoints (CYAN)
            # =========================
            for px, py in points:
                cv2.circle(annotated_frame,(px,py),4,(255,255,0),-1)

            # =========================
            # Highlight fingertip
            # =========================
            fingers_to_check = [4,8,12,16,20]

            for fid in fingers_to_check:

                fcx, fcy = points[fid]

                cv2.circle(annotated_frame,(fcx,fcy),5,(255,0,0),-1)

                # if xmin <= fcx <= xmax and ymin <= fcy <= ymax:
                #     ada_di_bbox = True

            # =========================
            # Draw hand label
            # =========================
            hx, hy = points[0]

            text = f"{label} {score:.2f}"

            (tw, th), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            cv2.rectangle(
                annotated_frame,
                (hx, hy - th - 10),
                (hx + tw, hy),
                (0,255,0),
                -1
            )

            cv2.putText(
                annotated_frame,
                text,
                (hx, hy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2
            )
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
                annotated_frame,
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
                annotated_frame,
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
                annotated_frame,
                (x1, y1 - h - 10),
                (x1 + w, y1),
                (0,255,0),
                -1
            )

            # -----------------------------
            # Draw label text (WHITE)
            # -----------------------------
            cv2.putText(
                annotated_frame,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2
            )

    cv2.imshow("Original Frame", frame)
    cv2.imshow("OpenVINO YOLO", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()