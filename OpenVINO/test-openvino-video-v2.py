from pathlib import Path
import cv2
from ultralytics import YOLO
from utils.check_device import OpenVINODeviceManager
import numpy as np

# =========================
# Model
# =========================
model_name = "yolo12n"
pt_model_path = f"{model_name}.pt"

det_model = YOLO(pt_model_path)
# =========================
# Export OpenVINO model
# =========================
det_model_path = Path(f"{model_name}_openvino_model/")

if not det_model_path.exists():
    print("Exporting model to OpenVINO...")
    det_model.export(format="openvino", dynamic=True, half=True, nms=True)

# =========================
# Check OpenVINO devices
# =========================
device_manager = OpenVINODeviceManager()
device = device_manager.get_best_device()

print("Using device:", device)

# =========================
# Load OpenVINO model
# =========================
import torch
import openvino as ov

core = ov.Core()
ov_model = YOLO(det_model_path, task="detect")

# =========================
# Video Source
# =========================
# 0 = webcam
# bisa diganti RTSP / file
video_source = 0

cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Cannot open video source")

# =========================
# Inference Loop
# =========================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    res = ov_model(
        source=frame,
        conf=0.25,
        iou=0.7,
        max_det=80,
        device=device,
        verbose=True
    )

    annotated = res[0].plot()

    cv2.imshow("YOLO OpenVINO Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# Cleanup
# =========================
cap.release()
cv2.destroyAllWindows()