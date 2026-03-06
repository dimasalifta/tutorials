from pathlib import Path
import urllib.request
import cv2
from ultralytics import YOLO
from utils.check_device import OpenVINODeviceManager
    
# =========================
# Download Image
# =========================
image_source = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg"

urllib.request.urlretrieve(image_source, "image.jpg")
print("Download selesai")

img = cv2.imread("image.jpg")

# =========================
# Model
# =========================
model_name = "yolo12n"
pt_model_path = f"{model_name}.pt"

pt_model = YOLO(pt_model_path)

# =========================
# Export OpenVINO model
# =========================
ov_model_path = Path(f"{model_name}_openvino_model/")

if not ov_model_path.exists():
    print("Exporting model to OpenVINO...")
    pt_model.export(format="openvino", dynamic=True, half=True)

# =========================
# Check OpenVINO devices
# =========================
device_manager = OpenVINODeviceManager()

device = device_manager.get_best_device()

# =========================
# Load OpenVINO model
# =========================
ov_model = YOLO(ov_model_path, task="detect")

# =========================
# Inference
# =========================
res = ov_model(
    source=img,
    conf=0.25,
    iou=0.7,
    max_det=80,
    device=device,
    verbose=False
)

# =========================
# Display
# =========================
cv2.imshow("Original Image", img)
cv2.imshow("Detection", res[0].plot())

cv2.waitKey(0)
cv2.destroyAllWindows()