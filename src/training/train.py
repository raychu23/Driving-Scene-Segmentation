import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os

from src.pipeline.pair_paths import get_pairs
from src.pipeline.dataset import BDD100KSegDataset
from src.training.models import get_deeplab, get_fastscnn, get_bisenetv2, get_baselinecnn, get_mobilenet 


# Pick model by name
def build_model(name, num_classes=19):
    name = name.lower()
    if name == "deeplab":
        return get_deeplab(num_classes)
    elif name == "fastscnn":
        return get_fastscnn(num_classes)
    elif name == "bisenetv2":
        return get_bisenetv2(num_classes)
    elif name == "baselinecnn":
        return get_baselinecnn(num_classes)
    elif name == "mobilenet":
        return get_mobilenet(num_classes)
    # elif name == "icnet":
    #     return get_icnet(num_classes)
    else:
        raise ValueError("Unknown model name")

def main(model_name="deeplab"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    train_pairs = get_pairs("data/bdd100k-seg/images/train",
                            "data/bdd100k-seg/labels/train")
    val_pairs   = get_pairs("data/bdd100k-seg/images/val",
                            "data/bdd100k-seg/labels/val")

    train_ds = BDD100KSegDataset(train_pairs, image_size=(512, 512))
    val_ds   = BDD100KSegDataset(val_pairs,   image_size=(512, 512))

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(model_name, num_classes=19).to(device)
    print(f"Training model: {model_name}")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 5

    train_history = []
    val_history = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)["out"]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                outputs = model(imgs)["out"]
                val_loss += criterion(outputs, masks).item()
        val_loss /= len(val_loader)

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | train {train_loss:.4f} | val {val_loss:.4f}")

    # save model
    torch.save(model.state_dict(), f"{model_name}_bdd100k.pth")
    print(f"Saved: {model_name}_bdd100k.pth")

    # save training curves
    os.makedirs("logs", exist_ok=True)
    hist = {
        "model": model_name,
        "train_loss": train_history,
        "val_loss": val_history
    }
    with open(os.path.join("logs", f"{model_name}_history.json"), "w") as f:
        json.dump(hist, f, indent=2)

#   python -m src.training.train deeplab
#   python -m src.training.train fastscnn
#   python -m src.training.train bisenetv2
#   python -m src.training.train baselinecnn
#   python -m src.training.train mobilenet
#   python -m src.training.train icnet
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "deeplab"   # default
    
    main(model_name)
