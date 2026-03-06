import openvino as ov


class OpenVINODeviceManager:

    def __init__(self):
        self.core = ov.Core()
        self.devices = self.core.available_devices

    def list_devices(self):
        """Print available OpenVINO devices"""
        print("Available OpenVINO devices:")
        for d in self.devices:
            print(f" - {d}")

    def get_best_device(self):
        """
        Return best device with priority:
        GPU -> CPU
        """

        if "GPU" in self.devices:
            device = "intel:gpu"
        elif "CPU" in self.devices:
            device = "intel:cpu"
        else:
            raise RuntimeError("No OpenVINO device available")

        return device

    def print_selected_device(self):
        device = self.get_best_device()
        print("Using device:", device)
        return device