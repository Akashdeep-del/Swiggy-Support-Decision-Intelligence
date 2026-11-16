"""Part 4: split quality, model voting, AdaBoost and network sizing."""
import math
import numpy as np
from common import (load_data, make_split, sigmoid, confusion,
                    fit_logistic, knn_probabilities, save_txt, save_csv)

X, y, value = load_data()
train, test = make_split(y)
raw_train = X[train]
y_train, y_test = y[train], y[test]

def entropy(labels):
    positive = float(np.mean(labels))
    if positive == 0 or positive == 1:
        return 0.0
    negative = 1 - positive
    return -(positive * math.log2(positive) + negative * math.log2(negative))

def gini(labels):
    positive = float(np.mean(labels))
    negative = 1 - positive
    return 1 - positive ** 2 - negative ** 2

root_entropy = entropy(y_train)
root_gini = gini(y_train)
conditions = {
    "S1": raw_train[:, 5] == 1,
    "S2": raw_train[:, 6] == 1,
    "S3": raw_train[:, 2] > 0.5,
    "S4": raw_train[:, 0] <= 24,
}
split_rows = []
for name, mask in conditions.items():
    yes, no = y_train[mask], y_train[~mask]
    information_gain = root_entropy - (
        len(yes) * entropy(yes) + len(no) * entropy(no)
    ) / len(y_train)
    gini_gain = root_gini - (
        len(yes) * gini(yes) + len(no) * gini(no)
    ) / len(y_train)
    split_rows.append([
        name, len(yes), int(yes.sum()), len(no), int(no.sum()),
        information_gain, gini_gain
    ])
best_split = max(split_rows, key=lambda row: row[5])[0]
best_gini_split = max(split_rows, key=lambda row: row[6])[0]
save_csv("part4_splits.csv",
         ["split", "n_yes", "positives_yes", "n_no", "positives_no",
          "info_gain", "gini_gain"], split_rows)

# Reuse the Part 2 split and training-only standardisation.
mean = X[train].mean(axis=0)
sd = X[train].std(axis=0, ddof=1)
Z = (X - mean) / sd
Z_train, Z_test = Z[train], Z[test]
weights, bias, _ = fit_logistic(Z_train, y_train)
logistic_probability = sigmoid(Z_test @ weights + bias)

# Recompute the same KNN cross-validation selection from Part 2.
cv_scores = []
positions = np.arange(len(train))
for k in range(1, 22, 2):
    scores = []
    for fold in range(5):
        valid = positions % 5 == fold
        probability = knn_probabilities(
            Z_train[~valid], y_train[~valid], Z_train[valid], k
        )
        prediction = (probability >= (k + 1) / (2 * k)).astype(int)
        scores.append(confusion(y_train[valid], prediction)["f1"])
    cv_scores.append([k, float(np.mean(scores))])
best_k = int(max(cv_scores, key=lambda row: (row[1], -row[0]))[0])
knn_probability = knn_probabilities(Z_train, y_train, Z_test, best_k)

# Stump probability is the positive share on the matching training side.
def split_side(rows):
    if best_split == "S1":
        return rows[:, 5] == 1
    if best_split == "S2":
        return rows[:, 6] == 1
    if best_split == "S3":
        return rows[:, 2] > 0.5
    return rows[:, 0] <= 24

train_side = split_side(X[train])
test_side = split_side(X[test])
stump_rate_yes = float(y_train[train_side].mean())
stump_rate_no = float(y_train[~train_side].mean())
stump_probability = np.where(test_side, stump_rate_yes, stump_rate_no)

logistic_prediction = (logistic_probability >= 0.3).astype(int)
knn_prediction = (knn_probability >= (best_k + 1) / (2 * best_k)).astype(int)
stump_prediction = (stump_probability >= 0.3).astype(int)
predictions = {
    "logistic": logistic_prediction,
    "knn": knn_prediction,
    "stump": stump_prediction,
    "hard_vote": ((logistic_prediction + knn_prediction + stump_prediction) >= 2).astype(int),
    "soft_vote": (((logistic_probability + knn_probability + stump_probability) / 3) >= 0.3).astype(int),
}
voting_rows = []
for name in ("logistic", "knn", "stump", "hard_vote", "soft_vote"):
    metrics = confusion(y_test, predictions[name])
    voting_rows.append([
        name, metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"],
        metrics["precision"], metrics["recall"], metrics["f1"]
    ])
save_csv("part4_voting.csv",
         ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"],
         voting_rows)

# AdaBoost: two rounds on the eight fixed examples.
true = np.array([1, 1, 1, -1, -1, -1, 1, -1])
h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])
weights_now = np.ones(8) / 8
adaboost_rows = []
alphas = []
errors = []
for round_number, hypothesis in enumerate((h1, h2), start=1):
    before = weights_now.copy()
    error = float(before[hypothesis != true].sum())
    alpha = 0.5 * math.log((1 - error) / error)
    unnormalised = before * np.exp(-alpha * true * hypothesis)
    weights_now = unnormalised / unnormalised.sum()
    errors.append(error)
    alphas.append(alpha)
    for point in range(8):
        adaboost_rows.append([
            round_number, point + 1, before[point],
            unnormalised[point], weights_now[point]
        ])
save_csv("part4_adaboost.csv",
         ["round", "point", "weight_before", "weight_unnormalised", "weight_after"],
         adaboost_rows)
ensemble_prediction = np.sign(alphas[0] * h1 + alphas[1] * h2)
ensemble_accuracy = float(np.mean(ensemble_prediction == true))

# Count parameters; no neural network is trained.
architecture = [
    ["mlp_hidden1", 784 * 16 + 16],
    ["mlp_hidden2", 16 * 16 + 16],
    ["mlp_output", 16 * 10 + 10],
    ["cnn_conv1", 5 * 5 * 1 * 6 + 6],
    ["cnn_conv2", 5 * 5 * 6 * 16 + 16],
    ["cnn_dense1", 5 * 5 * 16 * 32 + 32],
    ["cnn_dense2", 32 * 10 + 10],
]
save_csv("part4_architecture.csv", ["layer", "parameters"], architecture)

results = {
    "root_entropy": root_entropy,
    "root_gini": root_gini,
    "best_split_by_info_gain": best_split,
    "best_split_by_gini_gain": best_gini_split,
    "stump_rate_yes": stump_rate_yes,
    "stump_rate_no": stump_rate_no,
    "round1_error": errors[0],
    "round1_alpha": alphas[0],
    "round2_error": errors[1],
    "round2_alpha": alphas[1],
    "ensemble_accuracy": ensemble_accuracy,
    "oob_fraction": (1 - 1 / 560) ** 560,
    "mlp_total": sum(row[1] for row in architecture[:3]),
    "cnn_total": sum(row[1] for row in architecture[3:]),
    "fc_layer_params": 128 * 128 * 3 * 1024 + 1024,
    "conv_layer_params": 3 * 3 * 3 * 64 + 64,
    "rnn_total": 64 * 128 + 128 * 128 + 128 * 16 + 128 + 16,
}
save_txt("part4_results.txt", results)
print("Part 4 complete: five outputs created")
