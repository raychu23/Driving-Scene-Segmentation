# src/utils/plot_eda.py
import json
from collections import Counter
import matplotlib.pyplot as plt
import os
import sys

def load_metadata(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def count_categories(data):
    time_counts = Counter()
    weather_counts = Counter()
    scene_counts = Counter()

    for item in data:
        attrs = item.get("attributes", {})

        time = attrs.get("timeofday", "undefined")
        weather = attrs.get("weather", "undefined")
        scene = attrs.get("scene", "undefined")

        time_counts[time] += 1
        weather_counts[weather] += 1
        scene_counts[scene] += 1

    return time_counts, weather_counts, scene_counts


def plot_bar(counter, title, save_name):
    labels = list(counter.keys())
    values = [counter[k] for k in labels]

    plt.figure()
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("Count")
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(os.path.join("figures", save_name))
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.utils.plot_eda <json_path>")
        sys.exit(1)

    json_path = sys.argv[1]
    data = load_metadata(json_path)
    time_counts, weather_counts, scene_counts = count_categories(data)

    print("Time distribution:", time_counts)
    print("Weather distribution:", weather_counts)
    print("Scene distribution:", scene_counts)

    plot_bar(time_counts, "Time-of-day distribution (val)", "val_time_distribution.png")
    plot_bar(weather_counts, "Weather distribution (val)", "val_weather_distribution.png")
    plot_bar(scene_counts, "Scene distribution (val)", "val_scene_distribution.png")
