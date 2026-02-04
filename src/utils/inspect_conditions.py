import json
from collections import Counter
import sys

def inspect_conditions(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    times = Counter()
    weather = Counter()
    scenes = Counter()

    for item in data:
        attr = item["attributes"]
        times[attr["timeofday"]] += 1
        weather[attr["weather"]] += 1
        scenes[attr["scene"]] += 1

    print("Time distribution:", times)
    print("Weather distribution:", weather)
    print("Scene distribution:", scenes)

# python -m src.utils.inspect_conditions data/bdd100k-seg/weather/bdd100k_labels_images_val.json
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.utils.inspect_conditions <json_file>")
        sys.exit(1)

    json_path = sys.argv[1]
    inspect_conditions(json_path)