from pathlib import Path
import openvino as ov
from ultralytics import YOLO


class OpenVINODeviceManager:

    def __init__(self):
        self.core = ov.Core()
        self.devices = self.core.available_devices

    def list_devices(self):
        return self.devices

    def print_devices(self):

        print("Available OpenVINO devices:")

        for d in self.devices:
            print(f" - {d}")

    def get_best_device(self):
        """
        Priority:
        GPU -> CPU
        """

        if "GPU" in self.devices:
            return "intel:gpu"

        if "CPU" in self.devices:
            return "intel:cpu"

        raise RuntimeError("No OpenVINO device available")


class OpenVINOYOLODetector:

    def __init__(self, model_name="yolo11n", device=None):

        if ".pt" in model_name:
            model_name = model_name.split(".")[0]   
        
        self.model_name = model_name
        self.pt_model_path = f"{model_name}.pt"
        self.ov_model_dir = Path(f"{model_name}_openvino_model")

        self.device_manager = OpenVINODeviceManager()
        self.device_manager.print_devices()
        if device is None:
            self.device = self.device_manager.get_best_device()
        else:
            self.device = device

        self.model = None
        self._prepare_model()

    def _prepare_model(self):

        pt_model = YOLO(self.pt_model_path)

        if not self.ov_model_dir.exists():

            print("Exporting model to OpenVINO...")

            pt_model.export(
                format="openvino",
                dynamic=True,
                half=True,
                nms=True
            )

        self.model = YOLO(self.ov_model_dir, task="detect")

        print("Using device:", self.device)

    def predict(self, frame, conf=0.25, iou=0.7, max_det=80, verbose=True):

        results = self.model(
            source=frame,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=self.device,
            verbose=verbose
        )

        return results

    def plot(self, results):

        return results[0].plot()