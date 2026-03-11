import time
import cv2
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

# ================================
# Webcam
# ================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

# bbox contoh
bbox = (500, 200, 800, 500)

while cap.isOpened():

    ret, frame = cap.read()
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
    result = detector.detect_for_video(mp_image, timestamp_ms)

    ada_di_bbox = False

    # ================================
    # Draw bbox
    # ================================

    xmin, ymin, xmax, ymax = bbox
    cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(255,255,0),2)

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
    if result.hand_landmarks:

        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):

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

                cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)

            # =========================
            # Draw keypoints (CYAN)
            # =========================
            for px, py in points:
                cv2.circle(frame,(px,py),4,(255,255,0),-1)

            # =========================
            # Highlight fingertip
            # =========================
            fingers_to_check = [4,8,12,16,20]

            for fid in fingers_to_check:

                fcx, fcy = points[fid]

                cv2.circle(frame,(fcx,fcy),5,(255,0,0),-1)

                if xmin <= fcx <= xmax and ymin <= fcy <= ymax:
                    ada_di_bbox = True

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
                frame,
                (hx, hy - th - 10),
                (hx + tw, hy),
                (0,255,0),
                -1
            )

            cv2.putText(
                frame,
                text,
                (hx, hy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2
            )

    if ada_di_bbox:
        print("ADA")

    # ================================
    # show frame
    # ================================

    cv2.imshow("MediaPipe HandLandmarker", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()