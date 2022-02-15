import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

backbone = "vgg"
backbone_pretty = "VGG"
criteria = ["Two Structures", "Cystic Plate", "H. Triangle"]

r_path = f"results/{backbone}_test_results.csv"
model_results = pd.read_csv(r_path)

targets_c1 = model_results[f"Two Structures Label"]
probas_c1 = model_results[f"Two Structures Score"]
precisions_c1, recalls_c1, thresholds = precision_recall_curve(targets_c1, probas_c1)


targets_c2 = model_results[f"Cystic Plate Label"]
probas_c2 = model_results[f"Cystic Plate Score"]
precisions_c2, recalls_c2, thresholds = precision_recall_curve(targets_c2, probas_c2)

targets_c3 = model_results[f"H. Triangle Label"]
probas_c3 = model_results[f"H. Triangle Score"]
precisions_c3, recalls_c3, thresholds = precision_recall_curve(targets_c3, probas_c3)

init = 3000
plt.plot(recalls_c1, precisions_c1, color="turquoise", label="Two Structures")
plt.plot(recalls_c2, precisions_c2, color="tomato", label="Cystic Plate")
plt.plot(recalls_c3, precisions_c3, color="gold", label="Hep. Triangle")
plt.legend()
plt.title(backbone_pretty)
plt.grid(color='black', linestyle='--', linewidth=1, alpha=0.15)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.savefig(f"results/pr_curve_{backbone}.png")
