import os
import glob

# Step 1: 
# Make the pairs of images and masks
def get_pairs(image_dir, mask_dir):
    image_files = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg")) +
        glob.glob(os.path.join(image_dir, "*.png"))
    )

    pairs = []

    for img_path in image_files:
        fname = os.path.basename(img_path)
        base = os.path.splitext(fname)[0]   # removes .jpg or .png

        # Adjust to naming convention of masks ..._train_id.png
        mask_path = os.path.join(mask_dir, base + "_train_id.png")

        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
        else:
            print(f"[WARNING] Mask missing for {img_path}")

    return pairs

# Verify that the pairs are correct
if __name__ == "__main__":
    pairs = get_pairs(
        "data/bdd100k-seg/images/train",
        "data/bdd100k-seg/labels/train"
    )
    print("Total matched pairs:", len(pairs))
    if len(pairs) > 0:
        print(pairs[0])

