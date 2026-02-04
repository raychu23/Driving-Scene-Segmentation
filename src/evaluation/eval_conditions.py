import os
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, Subset

from src.training.models import (
    get_deeplab,
    get_fastscnn,
    get_mobilenet,
    get_baselinecnn,
)

from src.pipeline.bdd100k_with_attrs import BDD100KSegWithAttributes
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------
# Model factory
# ---------------------------------------------------------
MODEL_FACTORY = {
    "deeplab": get_deeplab,
    "fastscnn": get_fastscnn,
    "mobilenet": get_mobilenet,
    "baselinecnn": get_baselinecnn,
}

# ---------------------------------------------------------
# IoU helper
# ---------------------------------------------------------
def compute_iou(pred, mask, num_classes=19):
    ious = []
    pred = pred.view(-1)
    mask = mask.view(-1)

    for cls in range(num_classes):
        p = pred == cls
        m = mask == cls
        inter = (p & m).sum().item()
        union = (p | m).sum().item()
        if union > 0:
            ious.append(inter / union)

    return float(np.mean(ious)) if ious else float("nan")

# ---------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------
def evaluate_condition(model_name, weight_file, condition_name, condition_fn, dataset):

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    indices = []

    for idx in range(len(dataset)):
        attrs = dataset.metadata[idx]   # or whatever metadata field your class uses
        if condition_fn(attrs):
            indices.append(idx)

    print(
        f"[INFO] {model_name} | {condition_name}: "
        f"{len(indices)} samples"
    )

    if len(indices) == 0:
        return float("nan"), float("nan")

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )

    model = MODEL_FACTORY[model_name](num_classes=19)
    model.load_state_dict(torch.load(weight_file, map_location=device))
    model.to(device)
    model.eval()

    ious = []
    accs = []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)["out"]
            preds = outputs.argmax(dim=1)

            ious.append(compute_iou(preds.cpu(), masks.cpu()))
            accs.append((preds == masks).float().mean().item())

    return float(np.nanmean(ious)), float(np.mean(accs))

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    target_transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=Image.NEAREST),
    ])

    dataset = BDD100KSegWithAttributes(
        root="data/bdd100k-seg",
        split="val",
        transform=transform,
        target_transform=target_transform,
    )

    print(f"[INFO] Loaded validation set: {len(dataset)} images")
    # print(f"[INFO] Metadata entries: {len(dataset.scene_info)}")

    # -----------------------------------------------------
    # Conditions
    # -----------------------------------------------------
    def cond_all(attrs):
        return True

    def cond_night(attrs):
        return attrs.get("timeofday") == "night"

    BAD_WEATHER = {"rainy", "snowy", "foggy", "overcast", "partly cloudy"}

    def cond_adverse(attrs):
        return attrs.get("weather") in BAD_WEATHER

    conditions = {
        "all": cond_all,
        "night": cond_night,
        "adverse": cond_adverse,
    }

    models = ["deeplab", "fastscnn", "mobilenet", "baselinecnn"]
    results = defaultdict(dict)

    for model_name in models:
        weight_file = f"{model_name}_bdd100k.pth"

        if not os.path.exists(weight_file):
            print(f"[WARN] Missing weights: {weight_file}")
            continue

        for cname, cfn in conditions.items():
            miou, acc = evaluate_condition(
                model_name, weight_file, cname, cfn, dataset
            )
            results[model_name][cname] = (miou, acc)
            print(
                f"{model_name},{cname}: "
                f"mIoU={miou:.4f}, PixelAcc={acc:.4f}"
            )

    # -----------------------------------------------------
    # CSV output
    # -----------------------------------------------------
    print("\nModel,Condition,mIoU,PixelAcc")
    for model in models:
        for cname in conditions.keys():
            if cname in results[model]:
                miou, acc = results[model][cname]
                print(f"{model},{cname},{miou:.4f},{acc:.4f}")




# olldd



# import os
# import sys
# import torch
# import numpy as np
# from collections import defaultdict
# from torch.utils.data import DataLoader, Subset

# from data.bdd100k_dataset import BDD100KDataset   # <-- USE THE NEW DATASET CLASS
# from src.training.models import (
#     get_deeplab, get_fastscnn, get_mobilenet, get_baselinecnn
# )
# from torchvision import transforms
# from PIL import Image

# MODEL_FACTORY = {
#     "deeplab":      get_deeplab,
#     "fastscnn":     get_fastscnn,
#     "mobilenet":    get_mobilenet,
#     "baselinecnn":  get_baselinecnn,
# }

# # -------------------------
# # IoU helper
# # -------------------------
# def compute_iou(pred, mask, num_classes=19):
#     ious = []
#     for cls in range(num_classes):
#         p = (pred == cls)
#         m = (mask == cls)
#         inter = (p & m).sum()
#         union = (p | m).sum()
#         if union == 0:
#             continue
#         ious.append(float(inter) / float(union))
#     return float(np.mean(ious)) if ious else 0.0


# # -------------------------
# # EVALUATION CORE
# # -------------------------
# def evaluate_model(model_name, weight_file, cond_name, cond_fn, dataset):

#     device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#     subset_indices = []
#     missing_meta = 0

#     for idx, img_path in enumerate(dataset.image_filenames):
#         fname = img_path.name
#         attrs = dataset.scene_info.get(fname)

#         if attrs is None:
#             missing_meta += 1
#             continue

#         if cond_fn(attrs):
#             subset_indices.append(idx)

#     print(f"[INFO] {model_name} | {cond_name}: subset size = {len(subset_indices)} "
#           f"(missing metadata: {missing_meta})")

#     if len(subset_indices) == 0:
#         return 0.0, 0.0

#     loader = DataLoader(
#         Subset(dataset, subset_indices),
#         batch_size=1,
#         shuffle=False
#     )

#     model = MODEL_FACTORY[model_name](num_classes=19)
#     model.load_state_dict(torch.load(weight_file, map_location=device))
#     model.to(device)
#     model.eval()

#     ious, accs = [], []

#     with torch.no_grad():
#         for imgs, masks, _ in loader:
#             imgs, masks = imgs.to(device), masks.to(device)

#             out = model(imgs)["out"]
#             pred = out.argmax(dim=1)

#             ious.append(compute_iou(pred.cpu(), masks.cpu()))
#             accs.append((pred == masks).float().mean().item())

#     return float(np.mean(ious)), float(np.mean(accs))


# # -------------------------
# # MAIN SCRIPT ENTRY
# # -------------------------
# if __name__ == "__main__":

#     if len(sys.argv) != 2:
#         print("Usage: python -m src.evaluation.eval_conditions <bdd_root>")
#         sys.exit(1)

#     base_path = sys.argv[1]

#     transform = transforms.Compose([
#         transforms.Resize((512, 512)),
#         transforms.ToTensor()
#     ])

#     target_transform = transforms.Compose([
#         transforms.Resize((512, 512), interpolation=Image.NEAREST)
#     ])
    
#     # hi
#     from src.pipeline.bdd100k_with_attrs import BDD100KSegWithAttributes

#     transform = transforms.Compose([
#         transforms.Resize((512,512)),
#         transforms.ToTensor()
#     ])

#     target_transform = transforms.Compose([
#         transforms.Resize((512,512), interpolation=Image.NEAREST)
#     ])

#     dataset = BDD100KSegWithAttributes(
#         root="data/bdd100k-seg",
#         split="val",
#         transform=transform,
#         target_transform=target_transform
#     )
#     # hi

#     print(f"[INFO] Loaded dataset size: {len(dataset)}")
#     print(f"[INFO] Metadata entries: {len(dataset.scene_info)}")

#     # Conditions
#     def cond_all(attrs): return True
#     def cond_night(attrs): return attrs.get("timeofday") == "night"

#     BAD_WEATHER = {"rainy", "snowy", "foggy", "overcast", "partly cloudy"}
#     def cond_adverse(attrs): return attrs.get("weather") in BAD_WEATHER

#     conditions = {
#         "all": cond_all,
#         "night": cond_night,
#         "adverse": cond_adverse,
#     }

#     models = ["deeplab", "fastscnn", "mobilenet", "baselinecnn"]
#     results = defaultdict(dict)

#     for model_name in models:
#         weight_file = f"{model_name}_bdd100k.pth"
#         if not os.path.exists(weight_file):
#             print(f"[WARN] Missing weights: {weight_file}, skipping")
#             continue

#         for cname, cfn in conditions.items():
#             mIoU, acc = evaluate_model(model_name, weight_file, cname, cfn, dataset)
#             results[model_name][cname] = (mIoU, acc)
#             print(f"{model_name},{cname}: mIoU={mIoU:.3f}, acc={acc:.3f}")

#     print("\nModel,Condition,mIoU,PixelAcc")
#     for m in models:
#         for cname in ["all", "night", "adverse"]:
#             if cname in results[m]:
#                 iou, acc = results[m][cname]
#                 print(f"{m},{cname},{iou:.4f},{acc:.4f}")






# import json
# import os
# import sys
# from collections import defaultdict

# import torch
# import numpy as np
# from torch.utils.data import DataLoader, Subset

# from src.pipeline.dataset import BDD100KSegDataset
# from src.pipeline.pair_paths import get_pairs
# from src.training.models import (
#     get_deeplab, get_fastscnn, get_baselinecnn, get_mobilenet
# )

# # -------------------------
# # MODEL FACTORY
# # -------------------------
# MODEL_FACTORY = {
#     "deeplab":      get_deeplab,
#     "fastscnn":     get_fastscnn,
#     "mobilenet":    get_mobilenet,
#     "baselinecnn":  get_baselinecnn,
# }

# # -------------------------
# # KEY HELPERS
# # -------------------------
# def stem_key_from_name(name: str) -> str:
#     """
#     Use the FULL filename stem (without extension) as the key.

#     Example:
#         '0a0a0b1a-7c39d841.jpg' -> '0a0a0b1a-7c39d841'
#     """
#     base = os.path.basename(name)
#     stem, _ = os.path.splitext(base)
#     return stem


# def stem_key_from_path(path: str) -> str:
#     """
#     Same logic as above, but starting from a full filesystem path.
#     """
#     base = os.path.basename(path)
#     stem, _ = os.path.splitext(base)
#     return stem


# # -------------------------
# # LOAD JSON METADATA
# # -------------------------
# def load_metadata(train_json, val_json):
#     metadata = {}

#     for path in [train_json, val_json]:
#         if not os.path.exists(path):
#             print(f"[WARN] Missing JSON: {path}")
#             continue

#         print(f"[INFO] Loading metadata from {path}")
#         with open(path, "r") as f:
#             data = json.load(f)

#         for item in data:
#             # the JSON already stores the correct filename (no directory)
#             key = stem_key_from_name(item["name"])
#             attrs = item.get("attributes", {})

#             metadata[key] = {
#                 "timeofday": attrs.get("timeofday", "undefined"),
#                 "weather":   attrs.get("weather",   "undefined"),
#                 "scene":     attrs.get("scene",     "undefined"),
#             }

#     print(f"[INFO] Loaded metadata entries: {len(metadata)}")
#     return metadata


# # -------------------------
# # IoU COMPUTATION
# # -------------------------
# def compute_iou(pred, mask, num_classes=19):
#     ious = []
#     for cls in range(num_classes):
#         p = (pred == cls)
#         m = (mask == cls)
#         inter = (p & m).sum()
#         union = (p | m).sum()
#         if union == 0:
#             continue
#         ious.append(float(inter) / float(union))
#     return float(np.mean(ious)) if ious else 0.0


# # -------------------------
# # EVALUATE MODEL ON CONDITION SUBSET
# # -------------------------
# def evaluate_model(model_name, weight_file, cond_name, cond_fn, meta_map):

#     device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#     val_pairs = get_pairs(
#         "data/bdd100k-seg/images/val",
#         "data/bdd100k-seg/labels/val"
#     )

#     dataset = BDD100KSegDataset(val_pairs, image_size=(512, 512))

#     subset_indices = []
#     missing_meta = 0

#     for idx, (img_path, mask_path) in enumerate(val_pairs):
#         key = stem_key_from_path(img_path)  # <-- match JSON key exactly

#         attrs = meta_map.get(key)
#         if attrs is None:
#             missing_meta += 1
#             continue

#         if cond_fn(attrs):
#             subset_indices.append(idx)

#     print(
#         f"[INFO] {model_name}: subset[{cond_name}] size = {len(subset_indices)} "
#         f"(val images without metadata: {missing_meta})"
#     )

#     if len(subset_indices) == 0:
#         return 0.0, 0.0

#     loader = DataLoader(
#         Subset(dataset, subset_indices),
#         batch_size=1,
#         shuffle=False,
#     )

#     model = MODEL_FACTORY[model_name](num_classes=19)
#     model.load_state_dict(torch.load(weight_file, map_location=device))
#     model.to(device)
#     model.eval()

#     ious, accs = [], []

#     with torch.no_grad():
#         for imgs, masks in loader:
#             imgs, masks = imgs.to(device), masks.to(device)

#             out = model(imgs)["out"]
#             pred = out.argmax(dim=1)

#             ious.append(compute_iou(pred.cpu(), masks.cpu()))
#             accs.append((pred == masks).float().mean().item())

#     return float(np.mean(ious)), float(np.mean(accs))


# # -------------------------
# # MAIN SCRIPT
# # -------------------------
# if __name__ == "__main__":

#     if len(sys.argv) != 3:
#         print("Usage: python -m src.evaluation.eval_conditions <train_json> <val_json>")
#         sys.exit(1)

#     train_json = sys.argv[1]
#     val_json   = sys.argv[2]

#     meta_map = load_metadata(train_json, val_json)

#     # Conditions
#     def cond_all(attrs): return True
#     def cond_night(attrs): return attrs["timeofday"] == "night"

#     BAD_WEATHER = {"rainy", "snowy", "foggy", "overcast", "partly cloudy"}
#     def cond_adverse(attrs): return attrs["weather"] in BAD_WEATHER

#     conditions = {
#         "all": cond_all,
#         "night": cond_night,
#         "adverse": cond_adverse,
#     }

#     models = ["deeplab", "fastscnn", "mobilenet", "baselinecnn"]

#     results = defaultdict(dict)

#     for model_name in models:
#         weight_file = f"{model_name}_bdd100k.pth"
#         if not os.path.exists(weight_file):
#             print(f"[WARN] Missing weights: {weight_file}, skipping")
#             continue

#         for cname, cfn in conditions.items():
#             mIoU, acc = evaluate_model(model_name, weight_file, cname, cfn, meta_map)
#             results[model_name][cname] = (mIoU, acc)
#             print(f"{model_name} | {cname}: mIoU={mIoU:.3f}, acc={acc:.3f}")

#     print("\nModel,Condition,mIoU,PixelAcc")
#     for model_name in models:
#         for cname in ["all", "night", "adverse"]:
#             if cname in results[model_name]:
#                 m, a = results[model_name][cname]
#                 print(f"{model_name},{cname},{m:.4f},{a:.4f}")


# import json
# import os
# import sys
# from collections import defaultdict

# import torch
# import numpy as np
# from torch.utils.data import DataLoader, Subset

# from src.pipeline.dataset import BDD100KSegDataset
# from src.pipeline.pair_paths import get_pairs
# from src.training.models import (
#     get_deeplab, get_fastscnn, get_baselinecnn, get_mobilenet
# )

# # -------------------------
# # MODEL FACTORY
# # -------------------------
# MODEL_FACTORY = {
#     "deeplab":      get_deeplab,
#     "fastscnn":     get_fastscnn,
#     "mobilenet":    get_mobilenet,
#     "baselinecnn":  get_baselinecnn,
# }

# # -------------------------
# # FILENAME PREFIX EXTRACTOR
# # -------------------------
# def prefix(name):
#     """
#     Extract the unique BDD100K ID prefix before the first hyphen.
#     Example:
#         '7d06fefd-f7be05a6.jpg' → '7d06fefd'
#         '7d06fefd-f7be5d44.jpg' → '7d06fefd'
#     """
#     return name.split("-")[0]


# # -------------------------
# # LOAD JSON METADATA
# # -------------------------
# def load_metadata(train_json, val_json):
#     metadata = {}

#     for path in [train_json, val_json]:
#         if not os.path.exists(path):
#             print(f"[WARN] Missing JSON: {path}")
#             continue

#         with open(path, "r") as f:
#             data = json.load(f)

#         for item in data:
#             name = item["name"]  # e.g. "7d06fefd-f7be05a6.jpg"
#             base = os.path.basename(name)
#             stem = os.path.splitext(base)[0]     # remove .jpg/.png
#             key = prefix(stem)                    # only KEEP ID prefix

#             attrs = item.get("attributes", {})

#             metadata[key] = {
#                 "timeofday": attrs.get("timeofday", "undefined"),
#                 "weather":   attrs.get("weather",   "undefined"),
#                 "scene":     attrs.get("scene",     "undefined"),
#             }

#     print(f"[INFO] Loaded metadata entries: {len(metadata)}")
#     return metadata


# # -------------------------
# # IoU COMPUTATION
# # -------------------------
# def compute_iou(pred, mask, num_classes=19):
#     ious = []
#     for cls in range(num_classes):
#         p = (pred == cls)
#         m = (mask == cls)
#         inter = (p & m).sum()
#         union = (p | m).sum()
#         if union == 0:
#             continue
#         ious.append(float(inter) / float(union))
#     return float(np.mean(ious)) if ious else 0.0


# # -------------------------
# # EVALUATE MODEL ON CONDITION SUBSET
# # -------------------------
# def evaluate_model(model_name, weight_file, cond_name, cond_fn, meta_map):

#     device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#     val_pairs = get_pairs(
#         "data/bdd100k-seg/images/val",
#         "data/bdd100k-seg/labels/val"
#     )

#     dataset = BDD100KSegDataset(val_pairs, image_size=(512, 512))

#     subset_indices = []
#     for idx, (img_path, mask_path) in enumerate(val_pairs):
#         base = os.path.basename(img_path)
#         stem = os.path.splitext(base)[0]   # remove .jpg/.png
#         key = prefix(stem)

#         attrs = meta_map.get(key, None)
#         if attrs is None:
#             continue

#         if cond_fn(attrs):
#             subset_indices.append(idx)

#     print(f"[INFO] {model_name}: subset[{cond_name}] size = {len(subset_indices)}")

#     if len(subset_indices) == 0:
#         return 0.0, 0.0

#     loader = DataLoader(Subset(dataset, subset_indices),
#                         batch_size=1, shuffle=False)

#     model = MODEL_FACTORY[model_name](num_classes=19)
#     model.load_state_dict(torch.load(weight_file, map_location=device))
#     model.to(device)
#     model.eval()

#     ious, accs = [], []

#     with torch.no_grad():
#         for imgs, masks in loader:
#             imgs, masks = imgs.to(device), masks.to(device)

#             out = model(imgs)["out"]
#             pred = out.argmax(dim=1)

#             ious.append(compute_iou(pred.cpu(), masks.cpu()))
#             accs.append((pred == masks).float().mean().item())

#     return float(np.mean(ious)), float(np.mean(accs))


# # -------------------------
# # MAIN SCRIPT
# # -------------------------
# if __name__ == "__main__":

#     if len(sys.argv) != 3:
#         print("Usage: python -m src.evaluation.eval_conditions <train_json> <val_json>")
#         sys.exit(1)

#     train_json = sys.argv[1]
#     val_json   = sys.argv[2]

#     meta_map = load_metadata(train_json, val_json)

#     # Conditions
#     def cond_all(attrs): return True
#     def cond_night(attrs): return attrs["timeofday"] == "night"

#     BAD_WEATHER = {"rainy", "snowy", "foggy", "overcast", "partly cloudy"}
#     def cond_adverse(attrs): return attrs["weather"] in BAD_WEATHER

#     conditions = {
#         "all": cond_all,
#         "night": cond_night,
#         "adverse": cond_adverse,
#     }

#     models = ["deeplab", "fastscnn", "mobilenet", "baselinecnn"]

#     results = defaultdict(dict)

#     for model_name in models:
#         weight_file = f"{model_name}_bdd100k.pth"
#         if not os.path.exists(weight_file):
#             print(f"[WARN] Missing weights: {weight_file}, skipping")
#             continue

#         for cname, cfn in conditions.items():
#             mIoU, acc = evaluate_model(model_name, weight_file, cname, cfn, meta_map)
#             results[model_name][cname] = (mIoU, acc)
#             print(f"{model_name} | {cname}: mIoU={mIoU:.3f}, acc={acc:.3f}")

#     print("\nModel,Condition,mIoU,PixelAcc")
#     for model_name in models:
#         for cname in ["all", "night", "adverse"]:
#             if cname in results[model_name]:
#                 m, a = results[model_name][cname]
#                 print(f"{model_name},{cname},{m:.4f},{a:.4f}")





# python -m src.evaluation.eval_conditions \
#    data/bdd100k_labels_release/bdd100k_labels_images_train.json \
#    data/bdd100k_labels_release/bdd100k_labels_images_val.json

# python -m src.evaluation.eval_conditions \
#    data/bdd100k-seg/weather/bdd100k_labels_images_train.json \
#    data/bdd100k-seg/weather/bdd100k_labels_images_val.json


# (base) a1@MacBookPro Project % python -m json.tool data/bdd100k-seg/weather/bdd100k_labels_images_train.json

# Expecting property name enclosed in double quotes: line 10364150 column 3 (char 339738624)
# (base) a1@MacBookPro Project % sed -n '10364130,10364170p' data/bdd100k-seg/weather/bdd100k_labels_images_train.json

    #                             11.229786,
    #                             457.925734
    #                         ]
    #                     ],
    #                     "types": "LL",
    #                     "closed": false
    #                 }
    #             ],
    #             "id": 455066
    #         }
    #     ]
    # },
    # {
    #     "name": "29959e83-fedc5ddd.jpg",
    #     "attributes": {
    #         "weather": "overcast",
    #         "scene": "residential",
    #         "timeofday": "dawn/dusk"
    #     },
    #     "timestamp": 10000,
    #     "labels": [
    #         {
    #             "category": "traffic light",
    #             "attributes": {
    #                 "occluded": false,
    #                 "truncated": false,
    #                 "trafficLightColor": "green"
    #             },
    #             "manualShape": true,
    #             "manualAttributes": true,
    #             "box2d": {
    #                 "x1": 619.12383,
    #                 "y1": 189.89727,
    #                 "x2": 633.806609,
    #                 "y2": 230.030198
    #             },
    #             "id": 455067
    #         },
    #         {
    #             "category": "traffic light",
    #             "attributes": {

