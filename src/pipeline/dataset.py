import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

# ---------------------------------------------------------
# BDD100KSegDataset:
#   Loads one (image, mask) pair at a time.
#   Applies resizing + normalization for images.
#   Masks stay integer-valued for segmentation.
# ---------------------------------------------------------
class BDD100KSegDataset(Dataset):
    def __init__(self, pairs, image_size=(512, 512)):
        self.pairs = pairs # list of (img_path, mask_path)
        self.image_size = image_size # uniform size for CNN input

        # Transform pipeline for images
        self.img_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        # Mask resize only (keep labels integer)
        self.mask_resize = T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST)

    # Return number of samples
    def __len__(self):
        return len(self.pairs)

    # Load a single sample
    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # Load RGB image & raw mask (0,1,2 values)
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Apply transforms
        img = self.img_transform(img)
        mask = self.mask_resize(mask)

        # Convert mask to tensor of ints
        mask_np = np.array(mask, dtype="int64")
        mask_np[mask_np == 255] = -100 # change to default ignore label
        mask = torch.from_numpy(mask_np).long()

        return img, mask
