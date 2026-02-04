import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class ACDCSegDataset(Dataset):
    def __init__(self, root, condition, split, image_size=(512,512)):
        """
        root: data/acdc-seg
        condition: fog | rain | snow | night
        split: train | val
        """
        self.image_dir = os.path.join(root, "rgb", condition, split)
        self.mask_dir  = os.path.join(root, "gt",  condition, split)

        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.png")))

        self.img_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225])
        ])

        self.mask_resize = T.Resize(
            image_size, interpolation=T.InterpolationMode.NEAREST
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base = os.path.basename(img_path).replace("_rgb_anon.png", "")
        mask_path = os.path.join(
            self.mask_dir, base + "_gt_labelTrainIds.png"
        )

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        img = self.img_transform(img)
        mask = self.mask_resize(mask)

        mask = np.array(mask, dtype="int64")
        mask[mask == 255] = -100  # ignore label
        mask = torch.from_numpy(mask)

        return img, mask
