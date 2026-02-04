import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

from src.training.models import (
    get_deeplab,
    get_mobilenet,
    get_baselinecnn,
)

# -------------------------
# CONFIG
# -------------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

IMAGE_PATH = "data/bdd100k-seg/images/val/7dc08598-f42e2015.jpg"
MASK_PATH  = "data/bdd100k-seg/labels/val/7dc08598-f42e2015_train_id.png"

MODEL_WEIGHTS = {
    "DeepLabV3": "deeplab_bdd100k.pth",
    "MobileNet": "mobilenet_bdd100k.pth",
    "BaselineCNN": "baselinecnn_bdd100k.pth",
}

MODEL_FACTORY = {
    "DeepLabV3": get_deeplab,
    "MobileNet": get_mobilenet,
    "BaselineCNN": get_baselinecnn,
}

NUM_CLASSES = 19
IMG_SIZE = (512, 512)

# -------------------------
# TRANSFORMS
# -------------------------
img_transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

mask_transform = T.Compose([
    T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.NEAREST)
])

# -------------------------
# LOAD IMAGE + MASK
# -------------------------
img = Image.open(IMAGE_PATH).convert("RGB")
mask = Image.open(MASK_PATH)

img_t = img_transform(img).unsqueeze(0).to(DEVICE)
mask_t = torch.from_numpy(np.array(mask_transform(mask))).long()

# -------------------------
# VISUALIZATION FUNCTION
# -------------------------
def visualize(img, gt, preds, model_names):
    n = len(preds)
    fig, axs = plt.subplots(n, 3, figsize=(12, 4 * n))

    for i, (pred, name) in enumerate(zip(preds, model_names)):
        axs[i, 0].imshow(img)
        axs[i, 0].set_title("RGB Image")

        axs[i, 1].imshow(gt)
        axs[i, 1].set_title("Ground Truth")

        axs[i, 2].imshow(pred)
        axs[i, 2].set_title(f"{name} Prediction")

        for j in range(3):
            axs[i, j].axis("off")

    plt.tight_layout()
    plt.show()

# -------------------------
# RUN MODELS
# -------------------------
predictions = []
model_names = []

for name, weight_path in MODEL_WEIGHTS.items():
    if not os.path.exists(weight_path):
        print(f"[WARN] Missing weights: {weight_path}, skipping")
        continue

    print(f"Running {name}...")

    model = MODEL_FACTORY[name](num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        out = model(img_t)["out"]
        pred = out.argmax(dim=1).squeeze(0).cpu().numpy()

    predictions.append(pred)
    model_names.append(name)

# -------------------------
# SHOW RESULTS
# -------------------------
visualize(img, mask_t.numpy(), predictions, model_names)

# python -m src.utils.visualize_single_image.py