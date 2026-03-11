import cv2
import threading
import queue
import numpy as np
import asyncio

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamTrack
from av import VideoFrame

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from utils.openvino_utils import OpenVINOYOLODetector

# -------------------------
# MODEL
# -------------------------
model = OpenVINOYOLODetector("yolo12n.pt")

# -------------------------
# CAMERA + INFERENCE BUFFER
# -------------------------
frame_queue = queue.Queue(maxsize=5)
latest_frame = None

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
            confs = boxes.conf.cpu().numpy()
            cls  = boxes.cls.cpu().numpy()

            for box, score, c in zip(xyxy, confs, cls):
                x1, y1, x2, y2 = map(int, box)
                label = names[int(c)]
                text = f"{label} {score:.2f}"
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame, text, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        latest_frame = frame

# start camera + inference threads
threading.Thread(target=camera_worker, daemon=True).start()
threading.Thread(target=inference_worker, daemon=True).start()

# -------------------------
# FASTAPI + WEBRTC
# -------------------------
app = FastAPI()
pcs = set()

class Offer(BaseModel):
    sdp: str
    type: str

class VideoTrack(MediaStreamTrack):
    kind = "video"

    async def recv(self):
        global latest_frame
        pts, time_base = await self.next_timestamp()

        if latest_frame is None:
            # kirim frame kosong jika belum ready
            empty_frame = np.zeros((480,640,3), np.uint8)
            frame = VideoFrame.from_ndarray(empty_frame, format="bgr24")
        else:
            frame = VideoFrame.from_ndarray(latest_frame, format="bgr24")

        frame.pts = pts
        frame.time_base = time_base
        return frame

@app.post("/offer")
async def offer(offer: Offer):
    pc = RTCPeerConnection()
    pcs.add(pc)

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=offer.sdp, type=offer.type)
    )

    pc.addTrack(VideoTrack())

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })