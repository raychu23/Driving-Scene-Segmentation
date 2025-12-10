from torch.utils.data import DataLoader
from dataset import BDD100KSegDataset
from pair_paths import get_pairs


# ---------------------------------------------------------
# get_loaders:
#   Builds PyTorch DataLoaders for training and validation.
#   Handles batching, shuffling, and parallel workers.
# ---------------------------------------------------------
def get_loaders(batch_size=4, img_size=(512, 512)):
    
    # Retrieve (image, mask) file pairs
    train_pairs = get_pairs("data/bdd100k-seg/images/train",
                            "data/bdd100k-seg/labels/train")
    val_pairs   = get_pairs("data/bdd100k-seg/images/val",
                            "data/bdd100k-seg/labels/val")

    # Build datasets
    train_dataset = BDD100KSegDataset(train_pairs, image_size=img_size)
    val_dataset   = BDD100KSegDataset(val_pairs, image_size=img_size)

    # Create loaders (train shuffles, val does not)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader
