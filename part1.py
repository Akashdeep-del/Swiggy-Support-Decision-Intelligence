"""Part 1: estimation, inference and regression calculations."""
import math
import numpy as np
from common import load_data, save_txt

X, outcome, value = load_data()
n = len(value)
x = X[:, 3]
mean_value = value.mean()
sd_value = value.std(ddof=1)
se_value = sd_value / math.sqrt(n)

results = {
    "n": n,
    "mean_value": mean_value,
    "sd_value": sd_value,
    "se_value": se_value,
    "ci95_low": mean_value - 1.96 * se_value,
    "ci95_high": mean_value + 1.96 * se_value,
    "z_stat": (mean_value - 8.0) / se_value,
    "reject_h0": "yes" if abs((mean_value - 8.0) / se_value) > 1.96 else "no",
    "n_required": math.ceil((1.96 * sd_value / 0.5) ** 2),
}

# Closed-form simple regression.
slope = np.sum((x - x.mean()) * (value - mean_value)) / np.sum((x - x.mean()) ** 2)
intercept = mean_value - slope * x.mean()
fitted = intercept + slope * x
residual = value - fitted
tss = np.sum((value - mean_value) ** 2)
ess = np.sum((fitted - mean_value) ** 2)
rss = np.sum(residual ** 2)
results.update({
    "reg_slope": slope, "reg_intercept": intercept,
    "tss": tss, "ess": ess, "rss": rss,
    "tss_equals_ess_plus_rss": "yes" if abs(tss - ess - rss) < 1e-6 * tss else "no",
    "r2": 1 - rss / tss, "rmse": math.sqrt(rss / n),
    "mae": np.mean(np.abs(residual)),
})

# Gradient descent on the standardised predictor.
x_sd = x.std(ddof=1)
z = (x - x.mean()) / x_sd
m = 0.0
c = 0.0
def cost():
    return np.mean((value - (m * z + c)) ** 2)
costs = {0: cost()}
for epoch in range(1, 501):
    error = value - (m * z + c)
    grad_m = -2 * np.mean(z * error)
    grad_c = -2 * np.mean(error)
    m -= 0.1 * grad_m
    c -= 0.1 * grad_c
    if epoch in (1, 10, 100, 500):
        costs[epoch] = cost()
gd_slope = m / x_sd
gd_intercept = c - m * x.mean() / x_sd
results.update({
    "gd_J0": costs[0], "gd_J1": costs[1], "gd_J10": costs[10],
    "gd_J100": costs[100], "gd_J500": costs[500],
    "gd_slope": gd_slope, "gd_intercept": gd_intercept,
    "gd_slope_gap": abs(gd_slope - slope),
})

# Compare residual spread in lower and upper fitted-value halves.
order = np.argsort(fitted, kind="stable")
lower, upper = order[:400], order[400:]
low_sd = residual[lower].std(ddof=1)
high_sd = residual[upper].std(ddof=1)
results.update({
    "resid_sd_lower_raw": low_sd,
    "resid_sd_upper_raw": high_sd,
    "resid_sd_ratio_raw": high_sd / low_sd,
})

# Log-log simple regression.
log_x = np.log(x)
log_y = np.log(value)
log_slope = np.sum((log_x - log_x.mean()) * (log_y - log_y.mean())) / np.sum((log_x - log_x.mean()) ** 2)
log_intercept = log_y.mean() - log_slope * log_x.mean()
log_fit = log_intercept + log_slope * log_x
log_residual = log_y - log_fit
log_tss = np.sum((log_y - log_y.mean()) ** 2)
log_order = np.argsort(log_fit, kind="stable")
log_low = log_residual[log_order[:400]].std(ddof=1)
log_high = log_residual[log_order[400:]].std(ddof=1)
log_r2 = 1 - np.sum(log_residual ** 2) / log_tss
results.update({
    "loglog_intercept": log_intercept, "loglog_slope": log_slope,
    "loglog_r2": log_r2, "loglog_rmse": math.sqrt(np.mean(log_residual ** 2)),
    "resid_sd_lower_log": log_low, "resid_sd_upper_log": log_high,
    "resid_sd_ratio_log": log_high / log_low,
})

# Multiple regression normal equation in the order specified in the brief.
design = np.column_stack([
    np.ones(n), log_x, X[:, 0], X[:, 1], X[:, 2],
    X[:, 4], X[:, 5], X[:, 6]
])
beta = np.linalg.solve(design.T @ design, design.T @ log_y)
multiple_fit = design @ beta
multiple_r2 = 1 - np.sum((log_y - multiple_fit) ** 2) / log_tss
for index, coefficient in enumerate(beta):
    results[f"mr_beta_{index}"] = coefficient
results["mr_r2"] = multiple_r2
results["mr_adj_r2"] = 1 - (1 - multiple_r2) * (n - 1) / (n - 7 - 1)
results["loglog_adj_r2"] = 1 - (1 - log_r2) * (n - 1) / (n - 1 - 1)

save_txt("part1_results.txt", results)
print("Part 1 complete: outputs/part1_results.txt")
