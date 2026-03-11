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
label_map = det_model.model.names
# =========================
# Export OpenVINO model
# =========================
det_model_path = Path(f"{model_name}_openvino_model/{model_name}.xml")

if not det_model_path.exists():
    print("Exporting model to OpenVINO...")
    det_model.export(format="openvino", dynamic=True, half=True)

# =========================
# Check OpenVINO devices
# =========================
# device_manager = OpenVINODeviceManager()
# device = device_manager.get_best_device()

# print("Using device:", device)

# =========================
# Load OpenVINO model
# =========================
import torch
import openvino as ov

core = ov.Core()
# ov_model = YOLO(ov_model_path, task="detect")

det_ov_model = core.read_model(det_model_path)

device = "GPU"
ov_config = {}
if device != "CPU":
    det_ov_model.reshape({0: [1, 3, 640, 640]})
if "GPU" in device or ("AUTO" in device and "GPU" in core.available_devices):
    ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "NO"}
det_compiled_model = core.compile_model(det_ov_model, device, ov_config)
# force init predictor
det_model.predict(np.zeros((640,640,3), dtype=np.uint8))

def infer(*args):
    result = det_compiled_model(args)
    return torch.from_numpy(result[0])


det_model.predictor.inference = infer
det_model.predictor.model.pt = False

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

    res = det_model(
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