import asyncio
import argparse
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
import aiohttp
import cv2
from av import VideoFrame
from utils.openvino_utils import OpenVINOYOLODetector
import numpy as np

# YOLO
model = OpenVINOYOLODetector("yolo12n.pt")

class WebcamStream(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        ret, frame = self.cap.read()
        if not ret:
            # jika gagal baca frame, tunggu sebentar
            await asyncio.sleep(0.01)
            return await self.recv()

        # YOLO inference
        results = model.predict(frame, conf=0.25, iou=0.7, max_det=80, verbose=False)
        r = results[0]
        boxes = r.boxes
        names = r.names

        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            for box, score, c in zip(xyxy, confs, cls):
                x1, y1, x2, y2 = map(int, box)
                label = names[int(c)]
                text = f"{label} {score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Convert ke RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

async def run(server_url):
    pc = RTCPeerConnection()
    pc.addTrack(WebcamStream())

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    client_id = "camera_1"

    async with aiohttp.ClientSession() as session:
        async with session.post(server_url + "/offer", json={
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "client_id": client_id
        }) as resp:
            answer = await resp.json()

    await pc.setRemoteDescription(RTCSessionDescription(
        sdp=answer["sdp"],
        type=answer["type"]
    ))

    # keep alive
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    asyncio.run(run(args.server))