import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from codecarbon import EmissionsTracker
from torch.quantization import DeQuantStub, QuantStub

from .config import Config


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ExperimentTracker:
    def __init__(self, project_name: str):
        self.tracker = EmissionsTracker(
            project_name=project_name,
            output_dir=Config.RESULT_DIR,
            measure_power_secs=15,
            save_to_file=True,
            log_level="error",
        )
        self.start_time = 0.0

    def start(self):
        self.start_time = time.time()
        self.tracker.start()

    def stop(self):
        elapsed_min = (time.time() - self.start_time) / 60
        self.tracker.stop()
        energy_kwh = (
            self.tracker.final_emissions_data.energy_consumed
            if self.tracker.final_emissions_data
            else 0.0
        )
        return elapsed_min, energy_kwh


def log_to_csv(data: dict):
    os.makedirs(Config.RESULT_DIR, exist_ok=True)
    fields = [
        "timestamp", "model_name", "technique", "pruning_amount", "best_epoch",
        "val_accuracy", "test_accuracy", "total_time_min", "total_energy_kwh", "model_size_mb",
    ]
    exists = os.path.isfile(Config.LOG_FILE)
    with open(Config.LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(data)


def get_model_size_mb(model_path: str) -> float:
    return os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0.0


def apply_pruning(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """Apply global L1 unstructured pruning and remove pruning reparameterization."""
    parameters = [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=amount)
    for module, _ in parameters:
        prune.remove(module, "weight")
    return model


class QuantizedModelWrapper(nn.Module):
    def __init__(self, model_fp32):
        super().__init__()
        self.quant = QuantStub()
        self.model_fp32 = model_fp32
        self.dequant = DeQuantStub()

    def forward(self, x):
        return self.dequant(self.model_fp32(self.quant(x)))


def apply_quantization(model: nn.Module, calibration_loader) -> nn.Module:
    model.to("cpu").eval()
    quantized_model = QuantizedModelWrapper(model)
    backend = "fbgemm"
    quantized_model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(quantized_model, inplace=True)
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            if i >= 20:
                break
            quantized_model(images.cpu())
    torch.quantization.convert(quantized_model, inplace=True)
    return quantized_model
