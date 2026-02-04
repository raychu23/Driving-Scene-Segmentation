import json
import os
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch

def prefix(name):
    return name.split("-")[0]

class BDD100KSegWithAttributes(Dataset):
    def __init__(self, root, split="train", transform=None, target_transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform or transforms.ToTensor()
        self.target_transform = target_transform

        # -------------------------
        # PATHS MATCHING YOUR SYSTEM
        # -------------------------
        self.images_dir = self.root / "images" / split
        self.labels_dir = self.root / "labels" / split

        self.train_json = self.root / "weather" / "bdd100k_labels_images_train.json"
        self.val_json   = self.root / "weather" / "bdd100k_labels_images_val.json"

        # -------------------------
        # LOAD METADATA
        # -------------------------
        self.meta = self._load_metadata()

        # -------------------------
        # FILTER IMAGES THAT HAVE METADATA
        # -------------------------
        self.samples = self._collect_samples()

    def _load_metadata(self):
        meta = {}

        for path in [self.train_json, self.val_json]:
            if not path.exists():
                print(f"[WARN] metadata not found: {path}")
                continue

            with open(path, "r") as f:
                data = json.load(f)

            for item in data:
                name = item["name"]
                stem = os.path.splitext(name)[0]
                key = prefix(stem)

                attrs = item.get("attributes", {})

                meta[key] = {
                    "weather":   attrs.get("weather", "undefined"),
                    "scene":     attrs.get("scene", "undefined"),
                    "timeofday": attrs.get("timeofday", "undefined"),
                }

        print(f"[INFO] Loaded {len(meta)} metadata entries")
        return meta

    def _collect_samples(self):
        samples = []

        for fname in os.listdir(self.images_dir):
            if not fname.endswith(".jpg"):
                continue

            key = prefix(os.path.splitext(fname)[0])

            if key in self.meta:
                samples.append((self.images_dir / fname, self.labels_dir / fname.replace(".jpg", ".png"), self.meta[key]))

        print(f"[INFO] {self.split}: using {len(samples)} images with metadata")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, attrs = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = torch.tensor(np.array(mask), dtype=torch.long)

        return img, mask, attrs 