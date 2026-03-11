from nicegui import ui
import asyncio
import json
from aiortc import RTCPeerConnection, RTCSessionDescription
import aiohttp

SIGNALING_URL = "http://localhost:8000/offer"

async def start_webrtc(video_element):
    pc = RTCPeerConnection()

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            # pasang stream ke element NiceGUI
            video_element.srcObject = track._impl.streams[0]

    # buat offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # kirim offer ke FastAPI signaling
    async with aiohttp.ClientSession() as session:
        async with session.post(SIGNALING_URL, json={
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }) as resp:
            answer = await resp.json()

    # set remote description
    await pc.setRemoteDescription(RTCSessionDescription(
        sdp=answer["sdp"],
        type=answer["type"]
    ))

# --- UI ---
ui.label("YOLO Detection Dashboard").classes("text-h4")

# src wajib ada (bisa kosong string), jangan dihilangkan
video_el = ui.video(src="", autoplay=True, playsinline=True, style="width:800px").classes("my-video")

# jalankan coroutine di event loop NiceGUI
ui.run_task(lambda: start_webrtc(video_el))
ui.run()