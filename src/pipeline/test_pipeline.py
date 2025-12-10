import torch
import matplotlib.pyplot as plt

from .pair_paths import get_pairs
from .dataset import BDD100KSegDataset
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# 1) Test pairing logic
# ---------------------------------------------------------
pairs = get_pairs(
    "data/bdd100k-seg/images/train",
    "data/bdd100k-seg/labels/train"
)

print("Total pairs found:", len(pairs))
print("Example pair:", pairs[0])

# ---------------------------------------------------------
# 2) Test dataset loads a single sample
# ---------------------------------------------------------
dataset = BDD100KSegDataset(pairs, image_size=(256, 256))
img, mask = dataset[0]

print("Single image shape:", img.shape) # expected [3, H, W]
print("Single mask shape:", mask.shape) # expected [H, W]
print("Mask unique values:", mask.unique())

# ---------------------------------------------------------
# 3) Test DataLoader batching
# ---------------------------------------------------------
loader = DataLoader(dataset, batch_size=4, shuffle=False)
imgs, masks = next(iter(loader))

print("Batch image tensor:", imgs.shape) # expected [4, 3, H, W]
print("Batch mask tensor:", masks.shape) # expected [4, H, W]

# ---------------------------------------------------------
# 4) Visualize one sample to confirm correct alignment
# ---------------------------------------------------------
plt.subplot(1,2,1)
plt.imshow(img.permute(1,2,0)) # convert CHW → HWC
plt.title("Image")

plt.subplot(1,2,2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")

plt.tight_layout()
plt.show()
