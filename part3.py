"""Part 3: K-means segmentation, hierarchical clustering and PCA."""
import numpy as np
from common import load_data, save_txt, save_csv

X, y, value = load_data()
continuous = X[:, :5]
mean = continuous.mean(axis=0)
sd = continuous.std(axis=0, ddof=1)
Z = (continuous - mean) / sd

def run_kmeans(k):
    step = 800 // k
    centroids = Z[np.arange(k) * step].copy()
    previous = None
    while True:
        distances = np.sum((Z[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        if previous is not None and np.array_equal(labels, previous):
            break
        previous = labels.copy()
        for cluster in range(k):
            members = Z[labels == cluster]
            if len(members) > 0:
                centroids[cluster] = members.mean(axis=0)
    wcss = float(np.sum((Z - centroids[labels]) ** 2))
    return labels, centroids, wcss

results = {}
wcss = {}
for k in range(2, 7):
    _, _, wcss[k] = run_kmeans(k)
    results[f"wcss_k{k}"] = wcss[k]

elbow_k = 5
for k in (3, 4, 5):
    if wcss[k] - wcss[k + 1] < 0.5 * (wcss[k - 1] - wcss[k]):
        elbow_k = k
        break
results["elbow_k"] = elbow_k

# K=3 profiles, ordered by original-unit mean X3.
labels, centroids, _ = run_kmeans(3)
cluster_order = sorted(range(3), key=lambda j: continuous[labels == j, 2].mean())
segment_masks = {}
segment_rows = []
for position, cluster in enumerate(cluster_order):
    segment = chr(ord("A") + position)
    mask = labels == cluster
    segment_masks[segment] = mask
    results[f"size_segment_{segment}"] = int(mask.sum())
    segment_rows.append([
        segment, int(mask.sum()), *continuous[mask].mean(axis=0),
        float(y[mask].mean()), float(value[mask].mean())
    ])
save_csv("part3_segments.csv",
         ["segment", "size", "mean_X1", "mean_X2", "mean_X3",
          "mean_X4", "mean_X5", "outcome_rate", "mean_value"],
         segment_rows)

# Hierarchical clustering on the supplied six-cohort dissimilarity matrix.
D = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
])
hierarchical_rows = []
for method in ("single", "complete", "average"):
    clusters = [{i} for i in range(6)]
    heights = []
    for merge_number in range(1, 6):
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distances = [D[a, b] for a in clusters[i] for b in clusters[j]]
                if method == "single":
                    height = min(distances)
                elif method == "complete":
                    height = max(distances)
                else:
                    height = float(np.mean(distances))
                candidate = (height, min(clusters[i]), min(clusters[j]), i, j)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        height, _, _, i, j = best
        cluster_a, cluster_b = clusters[i], clusters[j]
        if min(cluster_a) > min(cluster_b):
            cluster_a, cluster_b = cluster_b, cluster_a
        def cluster_name(cluster):
            return "+".join(f"E{member + 1}" for member in sorted(cluster))
        hierarchical_rows.append([
            method, merge_number, cluster_name(cluster_a),
            cluster_name(cluster_b), height
        ])
        merged = cluster_a | cluster_b
        clusters = [
            cluster for index, cluster in enumerate(clusters)
            if index not in (i, j)
        ]
        clusters.append(merged)
        heights.append(height)
    gaps = np.diff(heights)
    largest_gap_index = int(np.argmax(gaps))
    merges_below_cut = largest_gap_index + 1
    results[f"{method}_clusters_at_cut"] = 6 - merges_below_cut
save_csv("part3_hierarchical.csv",
         ["linkage", "merge_no", "cluster_a", "cluster_b", "height"],
         hierarchical_rows)

# PCA from the covariance matrix.
covariance = (Z.T @ Z) / (len(Z) - 1)
eigenvalues, eigenvectors = np.linalg.eigh(covariance)
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]
for component in range(eigenvectors.shape[1]):
    largest = np.argmax(np.abs(eigenvectors[:, component]))
    if eigenvectors[largest, component] < 0:
        eigenvectors[:, component] *= -1
results["eigenvalue_sum"] = float(eigenvalues.sum())
for index in range(5):
    results[f"eigenvalue_{index + 1}"] = eigenvalues[index]
    results[f"explained_ratio_{index + 1}"] = eigenvalues[index] / eigenvalues.sum()
    results[f"cumulative_ratio_{index + 1}"] = eigenvalues[:index + 1].sum() / eigenvalues.sum()
results["components_for_80pct"] = int(np.argmax(np.cumsum(eigenvalues) / eigenvalues.sum() >= 0.8) + 1)
for index in range(5):
    results[f"pc1_loading_X{index + 1}"] = eigenvectors[index, 0]
    results[f"pc2_loading_X{index + 1}"] = eigenvectors[index, 1]
projection = Z @ eigenvectors[:, :2]
for row in range(3):
    results[f"row{row + 1}_pc1"] = projection[row, 0]
    results[f"row{row + 1}_pc2"] = projection[row, 1]
for segment, mask in segment_masks.items():
    results[f"mean_pc1_segment_{segment}"] = float(projection[mask, 0].mean())

save_txt("part3_results.txt", results)
print("Part 3 complete: three result files created")
