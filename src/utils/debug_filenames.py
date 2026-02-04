import json, os, sys

def debug(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    print("=== FIRST 10 JSON NAMES ===")
    for item in data[:10]:
        print(item["name"])

    img_dir = "data/bdd100k-seg/images/val"
    imgs = sorted(os.listdir(img_dir))[:10]

    print("\n=== FIRST 10 IMAGE FILENAMES (VAL DIR) ===")
    for x in imgs:
        print(x)

    print("\n=== FIRST 10 IMAGE STEMS (REMOVE JPG/PNG) ===")
    for x in imgs:
        print(os.path.splitext(x)[0])

# python -m src.utils.debug_filenames data/bdd100k-seg/weather/bdd100k_labels_images_val.json
if __name__ == "__main__":
    debug(sys.argv[1])
