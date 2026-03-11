import cv2
import threading
import queue
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from utils.openvino_utils import OpenVINOYOLODetector

app = FastAPI()

# -------------------------
# MODEL
# -------------------------

model = OpenVINOYOLODetector("yolo12n.pt")

# -------------------------
# GLOBAL BUFFER
# -------------------------

frame_queue = queue.Queue(maxsize=5)
latest_frame = None

# -------------------------
# CAMERA THREAD
# -------------------------

def camera_worker():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        if not frame_queue.full():
            frame_queue.put(frame)


# -------------------------
# INFERENCE THREAD
# -------------------------

def inference_worker():

    global latest_frame

    while True:

        frame = frame_queue.get()

        results = model.predict(
            frame,
            conf=0.25,
            iou=0.7,
            max_det=80,
            verbose=False
        )

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

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

        latest_frame = frame


# -------------------------
# STREAM GENERATOR
# -------------------------

def stream():

    global latest_frame

    while True:

        if latest_frame is None:
            continue

        ret, buffer = cv2.imencode(".jpg", latest_frame)

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )


# -------------------------
# API ENDPOINT
# -------------------------

@app.get("/video")
def video():

    return StreamingResponse(
        stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# -------------------------
# START THREADS
# -------------------------

threading.Thread(target=camera_worker, daemon=True).start()
threading.Thread(target=inference_worker, daemon=True).start()