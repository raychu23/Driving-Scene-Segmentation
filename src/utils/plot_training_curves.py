# src/utils/plot_training_curves.py
import json
import os
import sys
import matplotlib.pyplot as plt

def load_history(model_name):
    path = os.path.join("logs", f"{model_name}_history.json")
    with open(path, "r") as f:
        return json.load(f)

def plot_curves(model_names):
    # Plot training loss
    plt.figure()
    for name in model_names:
        hist = load_history(name)
        epochs = range(1, len(hist["train_loss"]) + 1)
        plt.plot(epochs, hist["train_loss"], label=f"{name} train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss vs epoch")
    plt.legend()
    plt.grid(True)
    os.makedirs("figures", exist_ok=True)
    plt.tight_layout()
    plt.savefig("figures/training_loss_curves.png")
    plt.close()

    # Plot validation loss
    plt.figure()
    for name in model_names:
        hist = load_history(name)
        epochs = range(1, len(hist["val_loss"]) + 1)
        plt.plot(epochs, hist["val_loss"], label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation loss vs epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/val_loss_curves.png")
    plt.close()


if __name__ == "__main__":
    # e.g. python -m src.utils.plot_training_curves deeplab bisenetv2 fastscnn
    if len(sys.argv) < 2:
        print("Usage: python -m src.utils.plot_training_curves <model1> [<model2> ...]")
        sys.exit(1)

    model_names = sys.argv[1:]
    plot_curves(model_names)
    plt.figure()
    for name in model_names:
        hist = load_history(name)
        epochs = range(1, len(hist["train_loss"]) + 1)
        plt.plot(epochs, hist["train_loss"], label=f"{name} train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss vs epoch")
    plt.legend()
    plt.grid(True)
    os.makedirs("figures", exist_ok=True)
    plt.tight_layout()
    plt.savefig("figures/training_loss_curves.png")
    plt.close()

    # Plot validation loss
    plt.figure()
    for name in model_names:
        hist = load_history(name)
        epochs = range(1, len(hist["val_loss"]) + 1)
        plt.plot(epochs, hist["val_loss"], label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation loss vs epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/val_loss_curves.png")
    plt.close()
