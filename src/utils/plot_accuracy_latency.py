import matplotlib.pyplot as plt

models = ["DeepLabV3", "MobileNet", "Baseline CNN"]
miou = [0.423, 0.388, 0.210]
latency = [92.4, 31.7, 14.9]  # replace with your measured values

plt.figure(figsize=(6,4))
plt.scatter(latency, miou)

for i, m in enumerate(models):
    plt.annotate(m, (latency[i], miou[i]), textcoords="offset points", xytext=(5,5))

plt.xlabel("Inference Latency (ms)")
plt.ylabel("mIoU")
plt.title("Accuracy vs Latency Tradeoff Across Models")
plt.grid(True)
plt.tight_layout()
plt.show()
