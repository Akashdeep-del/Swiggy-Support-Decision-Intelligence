"""Part 2: logistic regression, threshold evaluation and KNN."""
import math
import numpy as np
from common import (load_data, make_split, sigmoid, bce, confusion,
                    fit_logistic, knn_probabilities, save_txt, save_csv)

X, y, value = load_data()
train, test = make_split(y)
X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]

# Standardise using training rows only.
mean = X_train.mean(axis=0)
sd = X_train.std(axis=0, ddof=1)
Z = (X - mean) / sd
Z_train, Z_test = Z[train], Z[test]

weights, bias, history = fit_logistic(Z_train, y_train)
train_probability = sigmoid(Z_train @ weights + bias)
test_probability = sigmoid(Z_test @ weights + bias)

results = {
    "train_rows": len(train), "train_positives": int(y_train.sum()),
    "test_rows": len(test), "test_positives": int(y_test.sum()),
}
for epoch in (0, 1, 10, 100, 1000):
    results[f"loss_epoch_{epoch}"] = history[epoch]
for index, weight in enumerate(weights, start=1):
    results[f"weight_X{index}"] = weight
    results[f"odds_ratio_X{index}"] = math.exp(weight)
results["bias"] = bias
results["test_bce"] = bce(y_test, test_probability)

# Learning-rate comparison, restarted from zero each time.
for rate in (0.01, 0.1, 1.0):
    _, _, rate_history = fit_logistic(
        Z_train, y_train, rate=rate, steps=200, keep=(200,)
    )
    results[f"loss_epoch_200_eta_{rate}"] = rate_history[200]

for threshold in (0.5, 0.3):
    metrics = confusion(y_test, (test_probability >= threshold).astype(int))
    for key, number in metrics.items():
        results[f"tau_{threshold}_{key}"] = number
results["blind_accuracy"] = float(np.mean(y_test == 0))

threshold_rows = []
for threshold in np.arange(0.1, 1.0, 0.1):
    metrics = confusion(y_test, (test_probability >= threshold).astype(int))
    fpr = metrics["fp"] / (metrics["fp"] + metrics["tn"]) if metrics["fp"] + metrics["tn"] else 0
    threshold_rows.append([
        threshold, metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"],
        metrics["recall"], fpr
    ])
save_csv("part2_thresholds.csv",
         ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], threshold_rows)

# Five-fold CV: fold is the training-position modulo five.
cv_rows = []
for k in range(1, 22, 2):
    fold_f1 = []
    positions = np.arange(len(train))
    for fold in range(5):
        valid = positions % 5 == fold
        reference = ~valid
        probability = knn_probabilities(
            Z_train[reference], y_train[reference], Z_train[valid], k
        )
        prediction = (probability >= (k + 1) / (2 * k)).astype(int)
        fold_f1.append(confusion(y_train[valid], prediction)["f1"])
    cv_rows.append([k, float(np.mean(fold_f1))])
best_k = max(cv_rows, key=lambda row: (row[1], -row[0]))[0]
results["knn_best_k"] = int(best_k)
save_csv("part2_knn_cv.csv", ["K", "mean_cv_f1"], cv_rows)

knn_test_probability = knn_probabilities(
    Z_train, y_train, Z_test, int(best_k)
)
knn_metrics = confusion(
    y_test, (knn_test_probability >= (int(best_k) + 1) / (2 * int(best_k))).astype(int)
)
for key, number in knn_metrics.items():
    results[f"knn_{key}"] = number

# Select cost threshold using training predictions only.
training_costs = []
for threshold in (np.arange(1, 20) / 20):
    metrics = confusion(y_train, (train_probability >= threshold).astype(int))
    cost = 50 * metrics["fp"] + 400 * metrics["fn"]
    training_costs.append((cost, float(threshold)))
cost_train, cost_threshold = min(training_costs, key=lambda item: (item[0], item[1]))
test_at_cost = confusion(y_test, (test_probability >= cost_threshold).astype(int))
test_at_half = confusion(y_test, (test_probability >= 0.5).astype(int))
results["cost_tau"] = cost_threshold
results["cost_train_at_tau"] = cost_train
results["cost_test_at_tau"] = 50 * test_at_cost["fp"] + 400 * test_at_cost["fn"]
results["cost_test_at_0.5"] = 50 * test_at_half["fp"] + 400 * test_at_half["fn"]
results["cost_test_flag_nobody"] = 400 * int(y_test.sum())

save_txt("part2_results.txt", results)
print("Part 2 complete: outputs/part2_results.txt and two CSV files")
