#!/usr/bin/env python3
"""
=============================================================================
Unsupervised Learning - Clustering
=============================================================================
Based on: "Introduction to Machine Learning with Python" - Chapter 3 (pp.168-207)
Course: CSC582 - King Saud University

This notebook covers:
  1. k-Means Clustering
  2. Agglomerative Clustering  
  3. DBSCAN
  4. Comparing & Evaluating Clustering Algorithms
  
Enhanced with: Elbow Method, additional visualizations, and real-world dataset demo
=============================================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, ward

# Plot style
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print("All imports successful!")
print("=" * 70)


# =============================================================================
# SECTION 1: k-Means Clustering (Book pp.168-181)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: k-Means Clustering")
print("=" * 70)

# ── 1.1 Basic k-Means on Synthetic Data ─────────────────────────────────────
print("\n--- 1.1 Basic k-Means on Synthetic Blobs ---")

# Generate synthetic 2D data with 3 blobs
X, y_true = make_blobs(n_samples=300, centers=3, random_state=1, cluster_std=0.60)

# Apply k-Means
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
kmeans.fit(X)

print(f"Cluster labels (first 20): {kmeans.labels_[:20]}")
print(f"Cluster centers:\n{kmeans.cluster_centers_}")
print(f"Inertia (sum of squared distances): {kmeans.inertia_:.2f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original data
axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=40, alpha=0.7)
axes[0].set_title("Original Data (True Labels)", fontsize=14)
axes[0].set_xlabel("Feature 0")
axes[0].set_ylabel("Feature 1")

# k-Means result
axes[1].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', s=40, alpha=0.7)
axes[1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c='red', marker='^', s=200, edgecolors='black', linewidth=2,
                label='Cluster Centers')
axes[1].set_title("k-Means Clustering (k=3)", fontsize=14)
axes[1].set_xlabel("Feature 0")
axes[1].set_ylabel("Feature 1")
axes[1].legend()

plt.tight_layout()
plt.savefig("01_kmeans_basic.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 01_kmeans_basic.png")


# ── 1.2 Different Numbers of Clusters ───────────────────────────────────────
print("\n--- 1.2 Different Numbers of Clusters ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, k in zip(axes, [2, 3, 5]):
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    km.fit(X)
    ax.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap='viridis', s=40, alpha=0.7)
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
               c='red', marker='^', s=200, edgecolors='black', linewidth=2)
    ax.set_title(f"k-Means with k={k}", fontsize=14)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")

plt.tight_layout()
plt.savefig("02_kmeans_different_k.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 02_kmeans_different_k.png")


# ── 1.3 ENHANCEMENT: Elbow Method for Optimal k ─────────────────────────────
print("\n--- 1.3 [ENHANCEMENT] Elbow Method ---")

inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel("Number of Clusters (k)", fontsize=12)
axes[0].set_ylabel("Inertia (Within-cluster SSE)", fontsize=12)
axes[0].set_title("Elbow Method", fontsize=14)
axes[0].axvline(x=3, color='red', linestyle='--', alpha=0.7, label='Optimal k=3')
axes[0].legend()

# Silhouette plot
axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel("Number of Clusters (k)", fontsize=12)
axes[1].set_ylabel("Silhouette Score", fontsize=12)
axes[1].set_title("Silhouette Score vs k", fontsize=14)
axes[1].axvline(x=3, color='red', linestyle='--', alpha=0.7, label='Optimal k=3')
axes[1].legend()

plt.tight_layout()
plt.savefig("03_elbow_method.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 03_elbow_method.png")
print("The 'elbow' at k=3 confirms the optimal number of clusters.")


# ── 1.4 Failure Cases of k-Means ────────────────────────────────────────────
print("\n--- 1.4 Failure Cases of k-Means ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Case 1: Different densities
X_varied, y_varied = make_blobs(n_samples=200, cluster_std=[1.0, 2.5, 0.5],
                                 random_state=170)
y_pred = KMeans(n_clusters=3, random_state=0, n_init=10).fit_predict(X_varied)
axes[0].scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred, cmap='viridis', s=40)
axes[0].set_title("Failure: Different Densities", fontsize=13)
axes[0].set_xlabel("Feature 0")
axes[0].set_ylabel("Feature 1")

# Case 2: Elongated/non-spherical clusters
X_blob, y_blob = make_blobs(random_state=170, n_samples=600)
rng = np.random.RandomState(74)
transformation = rng.normal(size=(2, 2))
X_aniso = np.dot(X_blob, transformation)
y_pred2 = KMeans(n_clusters=3, random_state=0, n_init=10).fit_predict(X_aniso)
axes[1].scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred2, cmap='viridis', s=40)
axes[1].set_title("Failure: Non-Spherical Clusters", fontsize=13)
axes[1].set_xlabel("Feature 0")
axes[1].set_ylabel("Feature 1")

# Case 3: Two moons
X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=0)
y_pred3 = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(X_moons)
axes[2].scatter(X_moons[:, 0], X_moons[:, 1], c=y_pred3, cmap='viridis', s=40)
axes[2].set_title("Failure: Complex Shapes (Two Moons)", fontsize=13)
axes[2].set_xlabel("Feature 0")
axes[2].set_ylabel("Feature 1")

plt.suptitle("k-Means Failure Cases", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("04_kmeans_failures.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 04_kmeans_failures.png")


# ── 1.5 Vector Quantization: k-Means as Decomposition ───────────────────────
print("\n--- 1.5 Vector Quantization (k-Means as Decomposition) ---")

X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=0)
kmeans_10 = KMeans(n_clusters=10, random_state=0, n_init=10)
kmeans_10.fit(X_moons)
y_pred_10 = kmeans_10.predict(X_moons)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 10 clusters on two moons
axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_pred_10, s=60, cmap='Paired')
axes[0].scatter(kmeans_10.cluster_centers_[:, 0], kmeans_10.cluster_centers_[:, 1],
                s=100, marker='^', c=range(10), linewidth=2, cmap='Paired',
                edgecolors='black')
axes[0].set_title("k-Means with k=10 on Two Moons", fontsize=13)
axes[0].set_xlabel("Feature 0")
axes[0].set_ylabel("Feature 1")

# Distance features
distance_features = kmeans_10.transform(X_moons)
print(f"Distance feature shape: {distance_features.shape}")
print(f"  → Each point now has {distance_features.shape[1]} features")
print(f"  → (distance to each of the 10 cluster centers)")

# Show distance heatmap
im = axes[1].imshow(distance_features[:20], aspect='auto', cmap='YlOrRd')
axes[1].set_title("Distance Features (first 20 samples)", fontsize=13)
axes[1].set_xlabel("Cluster Center Index")
axes[1].set_ylabel("Sample Index")
plt.colorbar(im, ax=axes[1], label="Distance")

plt.tight_layout()
plt.savefig("05_vector_quantization.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 05_vector_quantization.png")


# =============================================================================
# SECTION 2: Agglomerative Clustering (Book pp.182-187)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: Agglomerative Clustering")
print("=" * 70)

# ── 2.1 Basic Agglomerative Clustering ──────────────────────────────────────
print("\n--- 2.1 Basic Agglomerative Clustering ---")

X_agg, y_agg = make_blobs(random_state=1)

agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X_agg)

plt.figure(figsize=(8, 6))
plt.scatter(X_agg[:, 0], X_agg[:, 1], c=assignment, cmap='viridis', s=60,
            edgecolors='black', linewidth=0.5)
plt.xlabel("Feature 0", fontsize=12)
plt.ylabel("Feature 1", fontsize=12)
plt.title("Agglomerative Clustering (Ward Linkage, 3 Clusters)", fontsize=14)
plt.savefig("06_agglomerative_basic.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 06_agglomerative_basic.png")


# ── 2.2 ENHANCEMENT: Comparing Linkage Methods ──────────────────────────────
print("\n--- 2.2 [ENHANCEMENT] Comparing Linkage Methods ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
linkage_methods = ['ward', 'average', 'complete']

for ax, linkage in zip(axes, linkage_methods):
    agg_link = AgglomerativeClustering(n_clusters=3, linkage=linkage)
    labels_link = agg_link.fit_predict(X_agg)
    ax.scatter(X_agg[:, 0], X_agg[:, 1], c=labels_link, cmap='viridis', s=60,
               edgecolors='black', linewidth=0.5)
    ax.set_title(f"Linkage: {linkage.capitalize()}", fontsize=14)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")

plt.suptitle("Agglomerative Clustering - Different Linkage Criteria", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("07_linkage_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 07_linkage_comparison.png")


# ── 2.3 Dendrograms ─────────────────────────────────────────────────────────
print("\n--- 2.3 Hierarchical Clustering & Dendrograms ---")

X_dendro, y_dendro = make_blobs(random_state=0, n_samples=12)

# Compute linkage
linkage_array = ward(X_dendro)

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linkage_array)

# Mark cut lines
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='red', linewidth=2)
ax.plot(bounds, [4, 4], '--', c='blue', linewidth=2)
ax.text(bounds[1], 7.25, '  2 clusters', va='center', fontsize=13, color='red')
ax.text(bounds[1], 4, '  3 clusters', va='center', fontsize=13, color='blue')

plt.xlabel("Sample Index", fontsize=12)
plt.ylabel("Cluster Distance (Ward)", fontsize=12)
plt.title("Dendrogram - Hierarchical Clustering", fontsize=14)
plt.savefig("08_dendrogram.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 08_dendrogram.png")
print("The dendrogram reads bottom-to-top: each merge shows which clusters joined.")
print("Longer branches = more dissimilar clusters being merged.")


# =============================================================================
# SECTION 3: DBSCAN (Book pp.187-190)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: DBSCAN (Density-Based Spatial Clustering)")
print("=" * 70)

# ── 3.1 DBSCAN Parameter Exploration ────────────────────────────────────────
print("\n--- 3.1 DBSCAN Parameter Exploration ---")

X_db, y_db = make_blobs(random_state=0, n_samples=12)

print("Effect of eps and min_samples on clustering:")
print(f"{'min_samples':>12} {'eps':>6} {'clusters':>40}")
print("-" * 62)

for min_s in [2, 3, 5]:
    for eps_val in [1.0, 1.5, 2.0, 3.0]:
        db = DBSCAN(min_samples=min_s, eps=eps_val)
        clusters = db.fit_predict(X_db)
        print(f"{min_s:>12} {eps_val:>6.1f} {str(clusters):>40}")

# ── 3.2 DBSCAN on Two Moons (where k-Means fails!) ─────────────────────────
print("\n--- 3.2 DBSCAN on Two Moons ---")

X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=0)

# Scale the data
scaler = StandardScaler()
X_moons_scaled = scaler.fit_transform(X_moons)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# k-Means (fails)
km_moons = KMeans(n_clusters=2, random_state=0, n_init=10)
labels_km = km_moons.fit_predict(X_moons_scaled)
axes[0].scatter(X_moons_scaled[:, 0], X_moons_scaled[:, 1], c=labels_km,
                cmap='viridis', s=60)
axes[0].set_title(f"k-Means (k=2)\nFails on complex shapes!", fontsize=13)
axes[0].set_xlabel("Feature 0")
axes[0].set_ylabel("Feature 1")

# Agglomerative (fails)
agg_moons = AgglomerativeClustering(n_clusters=2)
labels_agg = agg_moons.fit_predict(X_moons_scaled)
axes[1].scatter(X_moons_scaled[:, 0], X_moons_scaled[:, 1], c=labels_agg,
                cmap='viridis', s=60)
axes[1].set_title(f"Agglomerative (k=2)\nAlso fails!", fontsize=13)
axes[1].set_xlabel("Feature 0")
axes[1].set_ylabel("Feature 1")

# DBSCAN (succeeds!)
db_moons = DBSCAN(eps=0.5, min_samples=5)
labels_db = db_moons.fit_predict(X_moons_scaled)
axes[2].scatter(X_moons_scaled[:, 0], X_moons_scaled[:, 1], c=labels_db,
                cmap='viridis', s=60)
axes[2].set_title(f"DBSCAN (eps=0.5)\nSuccessfully separates moons!", fontsize=13)
axes[2].set_xlabel("Feature 0")
axes[2].set_ylabel("Feature 1")

plt.suptitle("Two Moons: DBSCAN vs. k-Means vs. Agglomerative", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("09_dbscan_two_moons.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 09_dbscan_two_moons.png")
print("DBSCAN correctly identifies complex non-convex cluster shapes!")


# ── 3.3 ENHANCEMENT: DBSCAN eps Sensitivity ─────────────────────────────────
print("\n--- 3.3 [ENHANCEMENT] DBSCAN eps Sensitivity ---")

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
eps_values = [0.1, 0.3, 0.5, 1.0]

for ax, eps_val in zip(axes, eps_values):
    db = DBSCAN(eps=eps_val, min_samples=5)
    labels = db.fit_predict(X_moons_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    ax.scatter(X_moons_scaled[:, 0], X_moons_scaled[:, 1], c=labels, cmap='viridis', s=40)
    ax.set_title(f"eps={eps_val}\n{n_clusters} clusters, {n_noise} noise", fontsize=12)
    ax.set_xlabel("Feature 0")

plt.suptitle("DBSCAN Sensitivity to eps Parameter", fontsize=15, y=1.05)
plt.tight_layout()
plt.savefig("10_dbscan_eps_sensitivity.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 10_dbscan_eps_sensitivity.png")


# =============================================================================
# SECTION 4: Comparing & Evaluating Clustering (Book pp.191-207)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: Comparing & Evaluating Clustering Algorithms")
print("=" * 70)

# ── 4.1 Evaluation WITH Ground Truth: ARI ───────────────────────────────────
print("\n--- 4.1 Adjusted Rand Index (ARI) - With Ground Truth ---")

X_eval, y_eval = make_moons(n_samples=200, noise=0.05, random_state=0)
scaler = StandardScaler()
X_eval_scaled = scaler.fit_transform(X_eval)

# Random assignment baseline
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X_eval))

algorithms = {
    "Random Assignment": random_clusters,
    "k-Means (k=2)": KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(X_eval_scaled),
    "Agglomerative (k=2)": AgglomerativeClustering(n_clusters=2).fit_predict(X_eval_scaled),
    "DBSCAN": DBSCAN().fit_predict(X_eval_scaled),
}

fig, axes = plt.subplots(1, 4, figsize=(20, 4))

for ax, (name, labels) in zip(axes, algorithms.items()):
    ari = adjusted_rand_score(y_eval, labels)
    ax.scatter(X_eval_scaled[:, 0], X_eval_scaled[:, 1], c=labels, cmap='viridis', s=40)
    ax.set_title(f"{name}\nARI: {ari:.2f}", fontsize=12)
    ax.set_xlabel("Feature 0")

plt.suptitle("Adjusted Rand Index (ARI) Comparison", fontsize=15, y=1.05)
plt.tight_layout()
plt.savefig("11_ari_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 11_ari_comparison.png")
print("\nARI Scores (1.0 = perfect, 0.0 = random):")
for name, labels in algorithms.items():
    ari = adjusted_rand_score(y_eval, labels)
    print(f"  {name:25s}: {ari:.2f}")


# ── 4.2 Why NOT to use accuracy_score ───────────────────────────────────────
print("\n--- 4.2 Why accuracy_score is Wrong for Clustering ---")

from sklearn.metrics import accuracy_score

clusters1 = [0, 0, 1, 1, 0]
clusters2 = [1, 1, 0, 0, 1]  # Same clustering, just labels swapped!

print(f"Clusters1: {clusters1}")
print(f"Clusters2: {clusters2}  (identical grouping, different labels)")
print(f"Accuracy:  {accuracy_score(clusters1, clusters2):.2f}  ← WRONG! Says 0%")
print(f"ARI:       {adjusted_rand_score(clusters1, clusters2):.2f}  ← CORRECT! Says 100%")
print("\nLesson: Cluster labels are arbitrary. Always use ARI or NMI, never accuracy!")


# ── 4.3 Evaluation WITHOUT Ground Truth: Silhouette Score ───────────────────
print("\n--- 4.3 Silhouette Score - Without Ground Truth ---")

fig, axes = plt.subplots(1, 4, figsize=(20, 4))

algorithms_sil = {
    "Random": random_clusters,
    "k-Means": KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(X_eval_scaled),
    "Agglomerative": AgglomerativeClustering(n_clusters=2).fit_predict(X_eval_scaled),
    "DBSCAN": DBSCAN().fit_predict(X_eval_scaled),
}

print("Silhouette Scores (higher = more compact clusters):")
for ax, (name, labels) in zip(axes, algorithms_sil.items()):
    # Silhouette requires at least 2 clusters
    n_unique = len(set(labels)) - (1 if -1 in labels else 0)
    if n_unique >= 2:
        sil = silhouette_score(X_eval_scaled, labels)
    else:
        sil = -1
    print(f"  {name:20s}: {sil:.2f}")
    ax.scatter(X_eval_scaled[:, 0], X_eval_scaled[:, 1], c=labels, cmap='viridis', s=40)
    ax.set_title(f"{name}\nSilhouette: {sil:.2f}", fontsize=12)

plt.suptitle("Silhouette Score Comparison", fontsize=15, y=1.05)
plt.tight_layout()
plt.savefig("12_silhouette_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 12_silhouette_comparison.png")
print("\nNote: k-Means gets HIGHER silhouette than DBSCAN here, even though")
print("DBSCAN's result is visually better! Silhouette favors compact spherical clusters.")


# =============================================================================
# SECTION 5: ENHANCEMENT - Real-World Dataset Demo (Iris)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: [ENHANCEMENT] Real-World Demo - Iris Dataset")
print("=" * 70)

from sklearn.datasets import load_iris

iris = load_iris()
X_iris = iris.data
y_iris = iris.target
feature_names = iris.feature_names

print(f"Iris dataset: {X_iris.shape[0]} samples, {X_iris.shape[1]} features")
print(f"Features: {feature_names}")
print(f"True classes: {iris.target_names}")

# Scale
scaler_iris = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)

# Apply all three algorithms
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# PCA for visualization (reduce to 2D)
pca_iris = PCA(n_components=2)
X_iris_2d = pca_iris.fit_transform(X_iris_scaled)

# True labels
axes[0, 0].scatter(X_iris_2d[:, 0], X_iris_2d[:, 1], c=y_iris, cmap='viridis', s=50,
                    edgecolors='black', linewidth=0.5)
axes[0, 0].set_title("True Labels", fontsize=14)

# k-Means
km_iris = KMeans(n_clusters=3, random_state=0, n_init=10)
labels_km_iris = km_iris.fit_predict(X_iris_scaled)
ari_km = adjusted_rand_score(y_iris, labels_km_iris)
axes[0, 1].scatter(X_iris_2d[:, 0], X_iris_2d[:, 1], c=labels_km_iris, cmap='viridis',
                    s=50, edgecolors='black', linewidth=0.5)
axes[0, 1].set_title(f"k-Means (k=3) — ARI: {ari_km:.2f}", fontsize=14)

# Agglomerative
agg_iris = AgglomerativeClustering(n_clusters=3)
labels_agg_iris = agg_iris.fit_predict(X_iris_scaled)
ari_agg = adjusted_rand_score(y_iris, labels_agg_iris)
axes[1, 0].scatter(X_iris_2d[:, 0], X_iris_2d[:, 1], c=labels_agg_iris, cmap='viridis',
                    s=50, edgecolors='black', linewidth=0.5)
axes[1, 0].set_title(f"Agglomerative — ARI: {ari_agg:.2f}", fontsize=14)

# DBSCAN
db_iris = DBSCAN(eps=0.9, min_samples=5)
labels_db_iris = db_iris.fit_predict(X_iris_scaled)
n_noise_iris = list(labels_db_iris).count(-1)
ari_db = adjusted_rand_score(y_iris, labels_db_iris)
axes[1, 1].scatter(X_iris_2d[:, 0], X_iris_2d[:, 1], c=labels_db_iris, cmap='viridis',
                    s=50, edgecolors='black', linewidth=0.5)
axes[1, 1].set_title(f"DBSCAN — ARI: {ari_db:.2f} ({n_noise_iris} noise pts)", fontsize=14)

for ax in axes.ravel():
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")

plt.suptitle("Clustering Algorithms on Iris Dataset (PCA-reduced)", fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig("13_iris_clustering.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 13_iris_clustering.png")

# Iris dendrogram
print("\n--- Iris Dataset Dendrogram ---")
linkage_iris = ward(X_iris_scaled)
plt.figure(figsize=(16, 6))
dendrogram(linkage_iris, truncate_mode='level', p=5, no_labels=True)
plt.xlabel("Sample Index", fontsize=12)
plt.ylabel("Cluster Distance", fontsize=12)
plt.title("Dendrogram of Iris Dataset (Agglomerative Clustering)", fontsize=14)
plt.savefig("14_iris_dendrogram.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 14_iris_dendrogram.png")


# =============================================================================
# SECTION 6: Summary Comparison Table
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: Summary & Algorithm Comparison")
print("=" * 70)

print("""
╔══════════════════════╦═══════════════════╦═══════════════════╦═══════════════════╗
║     Feature          ║     k-Means       ║  Agglomerative    ║      DBSCAN       ║
╠══════════════════════╬═══════════════════╬═══════════════════╬═══════════════════╣
║ Must set # clusters  ║       Yes         ║       Yes         ║        No         ║
║ Complex shapes       ║       No          ║       No          ║       Yes         ║
║ Noise detection      ║       No          ║       No          ║       Yes         ║
║ Scalability          ║    Excellent      ║      Good         ║      Good         ║
║ Cluster sizes        ║   Even-sized      ║  Even-sized       ║   Varies          ║
║ Interpretability     ║  Cluster centers  ║   Dendrogram      ║  Core/boundary    ║
║ Key parameters       ║     n_clusters    ║ n_clusters,       ║   eps,            ║
║                      ║                   ║ linkage           ║   min_samples     ║
╚══════════════════════╩═══════════════════╩═══════════════════╩═══════════════════╝
""")

print("=" * 70)
print("ALL SECTIONS COMPLETE!")
print("=" * 70)
print("\nGenerated plots:")
for i, name in enumerate([
    "01_kmeans_basic.png",
    "02_kmeans_different_k.png",
    "03_elbow_method.png",
    "04_kmeans_failures.png",
    "05_vector_quantization.png",
    "06_agglomerative_basic.png",
    "07_linkage_comparison.png",
    "08_dendrogram.png",
    "09_dbscan_two_moons.png",
    "10_dbscan_eps_sensitivity.png",
    "11_ari_comparison.png",
    "12_silhouette_comparison.png",
    "13_iris_clustering.png",
    "14_iris_dendrogram.png",
], start=1):
    print(f"  {i:2d}. {name}")
