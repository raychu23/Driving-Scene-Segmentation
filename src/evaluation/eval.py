import torch
import numpy as np
from torch.utils.data import DataLoader
from src.pipeline.dataset import BDD100KSegDataset
from src.pipeline.pair_paths import get_pairs
from src.training.models import     get_deeplab, get_fastscnn, get_bisenetv2, get_baselinecnn, get_mobilenet    

def compute_iou(pred, mask, num_classes=19):
    ious = []
    for cls in range(num_classes):
        pred_i = pred == cls
        mask_i = mask == cls
        inter = (pred_i & mask_i).sum()
        union = (pred_i | mask_i).sum()
        if union == 0:
            continue
        ious.append(float(inter) / float(union))
    return np.mean(ious)

def evaluate(model_name, weight_file):
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")

    val_pairs = get_pairs("data/bdd100k-seg/images/val",
                          "data/bdd100k-seg/labels/val")
    val_ds = BDD100KSegDataset(val_pairs)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = {
        "deeplab": get_deeplab,
        "fastscnn": get_fastscnn,
        "bisenetv2": get_bisenetv2,
        "baselinecnn": get_baselinecnn,
        "mobilenet": get_mobilenet
    }[model_name](num_classes=19)
    model.load_state_dict(torch.load(weight_file, map_location=device))
    model.to(device)
    model.eval()

    mean_iou = []
    pixel_acc = []

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)["out"]
            pred = out.argmax(dim=1)

            mean_iou.append(compute_iou(pred.cpu(), masks.cpu()))
            pixel_acc.append((pred == masks).float().mean().item())

    print("Model:", model_name)
    print("mIoU:", sum(mean_iou)/len(mean_iou))
    print("Pixel Acc:", sum(pixel_acc)/len(pixel_acc))

# python -m src.evaluation.eval deeplab deeplab_bdd100k.pth
# python -m src.evaluation.eval fastscnn fastscnn_bdd100k.pth
# python -m src.evaluation.eval bisenetv2 bisenetv2_bdd100k.pth
# python -m src.evaluation.eval baselinecnn baselinecnn_bdd100k.pth
# python -m src.evaluation.eval mobilenet mobilenet_bdd100k.pth
if __name__ == "__main__":
    # evaluate("deeplab", "deeplab_bdd100k.pth")
    import sys
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "baselinecnn"   # default

    evaluate(model_name, model_name + "_bdd100k.pth")

# Convert any model to ONNX
# python -m src.export.to_onnx deeplab deeplab.onnx

# Benchmark any model
# python -m src.benchmark.benchmark deeplab deeplab.onnx
# python -m src.benchmark.onnx_speed deeplab.onnx

# Visualize any model
# python -m src.visualization.visualize deeplab deeplab.onnx
# python -m src.visualization.visualize deeplab deeplab_bdd100k.pth

# Export any model
# python -m src.export.to_onnx deeplab deeplab.onnx
