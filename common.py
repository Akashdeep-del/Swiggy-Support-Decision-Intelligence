"""Small shared functions for the four assignment parts."""
from pathlib import Path
import csv
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "records.csv"
OUT_DIR = ROOT / "outputs"

def load_data():
    rows = np.genfromtxt(DATA_FILE, delimiter=",", skip_header=1, dtype=str)
    x = rows[:, 1:8].astype(float)
    y = rows[:, 8].astype(int)
    value = rows[:, 9].astype(float)
    return x, y, value

def save_txt(filename, values):
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / filename, "w", encoding="utf-8") as file:
        for key, value in values.items():
            if isinstance(value, str):
                shown = value
            elif isinstance(value, (int, np.integer)):
                shown = str(int(value))
            else:
                shown = f"{float(value):.4f}"
            file.write(f"{key} = {shown}\n")

def save_csv(filename, header, rows):
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in rows:
            writer.writerow([f"{x:.4f}" if isinstance(x, (float, np.floating)) else x for x in row])

def make_split(y):
    zero = np.where(y == 0)[0]
    one = np.where(y == 1)[0]
    train = np.sort(np.concatenate((zero[:7 * len(zero) // 10],
                                    one[:7 * len(one) // 10])))
    test = np.array([i for i in range(len(y)) if i not in set(train)])
    return train, test

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))

def bce(y, probability):
    p = np.clip(probability, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def confusion(y, prediction):
    tp = int(np.sum((y == 1) & (prediction == 1)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    tn = int(np.sum((y == 0) & (prediction == 0)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": (tp + tn) / len(y), "precision": precision,
            "recall": recall, "f1": f1,
            "specificity": tn / (tn + fp) if tn + fp else 0.0}

def fit_logistic(x, y, rate=0.1, steps=1000, keep=(0, 1, 10, 100, 200, 1000)):
    weights = np.zeros(x.shape[1])
    bias = 0.0
    history = {}
    if 0 in keep:
        history[0] = bce(y, sigmoid(x @ weights + bias))
    for epoch in range(1, steps + 1):
        probability = sigmoid(x @ weights + bias)
        error = probability - y
        weights -= rate * (x.T @ error) / len(y)
        bias -= rate * np.mean(error)
        if epoch in keep:
            history[epoch] = bce(y, sigmoid(x @ weights + bias))
    return weights, bias, history

def knn_probabilities(reference_x, reference_y, query_x, k):
    probabilities = []
    for query in query_x:
        distances = np.sum((reference_x - query) ** 2, axis=1)
        nearest = np.argsort(distances, kind="stable")[:k]
        probabilities.append(np.mean(reference_y[nearest]))
    return np.asarray(probabilities)
