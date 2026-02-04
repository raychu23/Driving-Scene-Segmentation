import torch
import numpy as np
from torch.utils.data import DataLoader
from src.pipeline.acdc_dataset import ACDCSegDataset
from src.training.models import get_deeplab, get_fastscnn, get_mobilenet, get_baselinecnn
import sys

MODEL_FACTORY = {
    "deeplab": get_deeplab,
    "fastscnn": get_fastscnn,
    "mobilenet": get_mobilenet,
    "baselinecnn": get_baselinecnn,
}

# def compute_iou(pred, mask, num_classes=19):
#     ious = []
#     for cls in range(num_classes):
#         p = (pred == cls)
#         m = (mask == cls)
#         inter = (p & m).sum()
#         union = (p | m).sum()
#         if union > 0:
#             ious.append((inter / union).item())
#     return np.mean(ious) if ious else 0.0

def compute_iou(pred, mask, num_classes=19, ignore_index=-100):
    ious = []
    for cls in range(num_classes):
        p = (pred == cls)
        m = (mask == cls)
        if m.sum() == 0:
            continue
        inter = (p & m).sum()
        union = (p | m).sum()
        if union > 0:
            ious.append(float(inter) / float(union))
    return float(np.mean(ious)) if len(ious) > 0 else 0.0

def eval_on_acdc(model_name, condition):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ds = ACDCSegDataset(
        root="data/acdc-seg",
        condition=condition,
        split="val"
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model = MODEL_FACTORY[model_name](num_classes=19)
    model.load_state_dict(torch.load(f"{model_name}_bdd100k.pth", map_location=device))
    model.to(device).eval()

    ious = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)["out"]
            print("output shape debug:", out.shape)  # should be [1, 19, H, W]
            pred = out.argmax(dim=1)
            ious.append(compute_iou(pred.cpu(), masks.cpu()))

    print(f"{model_name} | ACDC-{condition}: mIoU = {np.mean(ious):.3f}")

if __name__ == "__main__":
    # models = ["deeplab", "fastscnn", "mobilenet", "baselinecnn"]
    # conditions = ["fog", "rain", "snow", "night"]
    model_name = sys.argv[1]
    condition = sys.argv[2]
    eval_on_acdc(model_name, condition) 
    

# if __name__ == "__main__":
#     models = ["deeplab", "fastscnn", "mobilenet", "baselinecnn"]
#     conditions = ["fog", "rain", "snow", "night"]
#     for model_name in models:
#         for condition in conditions:
#             eval_on_acdc(model_name, condition) 
    