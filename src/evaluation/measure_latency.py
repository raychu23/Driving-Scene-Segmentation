import time
import torch
from src.training.models import (
    get_deeplab, get_mobilenet, get_baselinecnn
)

MODEL_FACTORY = {
    "deeplab": get_deeplab,
    "mobilenet": get_mobilenet,
    "baselinecnn": get_baselinecnn,
}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def measure_latency(model_name, runs=50):
    model = MODEL_FACTORY[model_name](num_classes=19).to(device)
    model.eval()

    dummy = torch.randn(1, 3, 512, 512).to(device)

    # warm-up
    with torch.no_grad():
        for _ in range(5):
            model(dummy)["out"]

    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.time()
            model(dummy)["out"]
            torch.mps.synchronize() if device.type == "mps" else None
            times.append((time.time() - start) * 1000)

    print(f"{model_name}: {sum(times)/len(times):.2f} ms")

if __name__ == "__main__":
    for m in ["deeplab", "mobilenet", "baselinecnn"]:
        measure_latency(m)
