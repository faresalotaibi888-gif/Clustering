"""
Unsupervised Learning — Clustering Algorithms
Course: CSC582 — Data Warehousing and Mining | King Saud University
Reference: Introduction to Machine Learning with Python — Chapter 3 (pp. 168–207)
GitHub: https://github.com/YOUR_USERNAME/ClusteringInDBSCAN
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_blobs, make_moons, load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score, silhouette_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, ward

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print("All imports successful!\n")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                            ║
# ║  SECTION 1 — K-MEANS CLUSTERING (pp. 168–181)                             ║
# ║  Assigned to: [Team Member Name]                                           ║
# ║                                                                            ║
# ║  TODO: Implement the following:                                            ║
# ║    1.1  Basic k-Means on make_blobs data (k=3)                            ║
# ║    1.2  Elbow Method — choosing optimal k (inertia + silhouette plot)      ║
# ║    1.3  k-Means failure cases (different densities, non-spherical, moons)  ║
# ║                                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

## ── Section 1: K-Means Clustering — TO BE FILLED BY TEAMMATE ──

# 1.1 Basic k-Means
# TODO: Generate make_blobs data, apply KMeans(n_clusters=3), plot original vs clustered

# 1.2 Elbow Method
# TODO: Loop k from 2 to 10, plot inertia and silhouette score vs k

# 1.3 k-Means Failure Cases
# TODO: Show 3 failure cases — different densities, elongated clusters, two moons


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                            ║
# ║  SECTION 2 — AGGLOMERATIVE CLUSTERING (pp. 182–187)                       ║
# ║  Assigned to: [Team Member Name]                                           ║
# ║                                                                            ║
# ║  TODO: Implement the following:                                            ║
# ║    2.1  Basic Agglomerative Clustering on make_blobs (3 clusters)          ║
# ║    2.2  Dendrogram visualization                                           ║
# ║    2.3  Linkage comparison (Ward vs Average vs Complete)                   ║
# ║                                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

## ── Section 2: Agglomerative Clustering — TO BE FILLED BY TEAMMATE ──

# 2.1 Basic Agglomerative Clustering
# TODO: Generate make_blobs data, apply AgglomerativeClustering(n_clusters=3), plot

# 2.2 Dendrogram
# TODO: Create dendrogram with ward linkage, show cut lines for 2 and 3 clusters

# 2.3 Linkage Comparison
# TODO: Compare Ward, Average, Complete linkage on same dataset


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                            ║
# ║  SECTION 3 — DBSCAN CLUSTERING (pp. 187–190)                              ║
# ║  Density-Based Spatial Clustering of Applications with Noise               ║
# ║                                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

## ── Section 3: DBSCAN Clustering ──

# ── 3.1 DBSCAN Parameter Exploration (Table) ──
print("=" * 65)
print("3.1 DBSCAN Parameter Exploration")
print("=" * 65)

X_db, y_db = make_blobs(random_state=0, n_samples=12)

print(f'{"min_samples":>12} {"eps":>6} {"clusters":>40}')
print('-' * 62)

for min_s in [2, 3, 5]:
    for eps_val in [1.0, 1.5, 2.0, 3.0]:
        db = DBSCAN(min_samples=min_s, eps=eps_val)
        clusters = db.fit_predict(X_db)
        print(f'{min_s:>12} {eps_val:>6.1f} {str(clusters):>40}')

print()


# ── 3.2 Effect of eps Parameter ──
print("3.2 Effect of eps Parameter")
X_moons, y_moons = make_moons(n_samples=300, noise=0.06, random_state=0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_moons)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
eps_values = [0.1, 0.2, 0.3, 0.5, 0.8, 1.5]

for ax, eps_val in zip(axes.ravel(), eps_values):
    db = DBSCAN(eps=eps_val, min_samples=5)
    labels = db.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    noise_mask = labels == -1
    ax.scatter(X_scaled[noise_mask, 0], X_scaled[noise_mask, 1],
               c='red', marker='x', s=50, label=f'Noise ({n_noise} pts)', zorder=3)

    cluster_mask = labels != -1
    if cluster_mask.any():
        ax.scatter(X_scaled[cluster_mask, 0], X_scaled[cluster_mask, 1],
                   c=labels[cluster_mask], cmap='viridis', s=40,
                   edgecolors='black', linewidth=0.3)

    if eps_val <= 0.5:
        circle = plt.Circle((X_scaled[150, 0], X_scaled[150, 1]),
                             eps_val, fill=False, color='red', linewidth=2,
                             linestyle='--', alpha=0.7)
        ax.add_patch(circle)
        ax.plot(X_scaled[150, 0], X_scaled[150, 1], 'r*', markersize=15, zorder=5)

    if n_clusters == 0: result, color = 'ALL NOISE!', 'red'
    elif n_clusters == 2: result, color = 'CORRECT!', 'green'
    elif n_clusters == 1: result, color = 'One big cluster', 'orange'
    else: result, color = 'Too fragmented', 'orange'

    ax.set_title(f'eps = {eps_val}\n{n_clusters} clusters, {n_noise} noise — {result}',
                 fontsize=13, fontweight='bold', color=color)
    ax.set_xlabel('Feature 0'); ax.set_ylabel('Feature 1')
    ax.legend(loc='upper right', fontsize=9)

fig.suptitle('EFFECT OF eps (min_samples fixed at 5)\n'
             'Red dashed circle = eps neighborhood radius',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plot_3_2_eps_effect.png', dpi=150, bbox_inches='tight')
plt.show()


# ── 3.3 Effect of min_samples Parameter ──
print("3.3 Effect of min_samples Parameter")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
min_samples_values = [2, 3, 5, 10, 20, 50]

for ax, min_s in zip(axes.ravel(), min_samples_values):
    db = DBSCAN(eps=0.5, min_samples=min_s)
    labels = db.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    core_mask = np.zeros(len(labels), dtype=bool)
    if hasattr(db, 'core_sample_indices_') and len(db.core_sample_indices_) > 0:
        core_mask[db.core_sample_indices_] = True

    noise_mask = labels == -1
    border_mask = (labels != -1) & (~core_mask)

    ax.scatter(X_scaled[noise_mask, 0], X_scaled[noise_mask, 1],
               c='red', marker='x', s=50, label=f'Noise ({n_noise})', zorder=3)
    if border_mask.any():
        ax.scatter(X_scaled[border_mask, 0], X_scaled[border_mask, 1],
                   c=labels[border_mask], cmap='viridis', s=30,
                   edgecolors='black', linewidth=0.3, alpha=0.6)
    if core_mask.any():
        ax.scatter(X_scaled[core_mask, 0], X_scaled[core_mask, 1],
                   c=labels[core_mask], cmap='viridis', s=60,
                   edgecolors='black', linewidth=0.5)

    n_core = core_mask.sum()
    n_border = border_mask.sum()
    ax.set_title(f'min_samples = {min_s}\n{n_clusters} clusters | '
                 f'Core: {n_core} | Border: {n_border} | Noise: {n_noise}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 0'); ax.set_ylabel('Feature 1')
    ax.legend(loc='upper right', fontsize=9)

fig.suptitle('EFFECT OF min_samples (eps fixed at 0.5)\n'
             'Large dots = Core | Small dots = Border | Red X = Noise',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plot_3_3_min_samples_effect.png', dpi=150, bbox_inches='tight')
plt.show()


# ── 3.4 DBSCAN vs k-Means vs Agglomerative on Two Moons ──
print("3.4 DBSCAN vs k-Means vs Agglomerative on Two Moons")

X_moons2, y_moons2 = make_moons(n_samples=200, noise=0.05, random_state=0)
scaler2 = StandardScaler()
X_moons2_scaled = scaler2.fit_transform(X_moons2)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

labels_km = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(X_moons2_scaled)
axes[0].scatter(X_moons2_scaled[:, 0], X_moons2_scaled[:, 1], c=labels_km, cmap='viridis', s=60)
axes[0].set_title('k-Means (k=2)\n❌ Fails on complex shapes!', fontsize=13)

labels_agg = AgglomerativeClustering(n_clusters=2).fit_predict(X_moons2_scaled)
axes[1].scatter(X_moons2_scaled[:, 0], X_moons2_scaled[:, 1], c=labels_agg, cmap='viridis', s=60)
axes[1].set_title('Agglomerative (k=2)\n❌ Also fails!', fontsize=13)

labels_db = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_moons2_scaled)
axes[2].scatter(X_moons2_scaled[:, 0], X_moons2_scaled[:, 1], c=labels_db, cmap='viridis', s=60)
axes[2].set_title('DBSCAN (eps=0.5)\n✅ Correctly separates!', fontsize=13)

for ax in axes:
    ax.set_xlabel('Feature 0'); ax.set_ylabel('Feature 1')

plt.suptitle('Two Moons: DBSCAN Succeeds Where Others Fail', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('plot_3_4_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# ── 3.5 DBSCAN Core / Border / Noise Visualization ──
print("3.5 DBSCAN Core / Border / Noise Visualization")

X_demo, y_demo = make_moons(n_samples=200, noise=0.12, random_state=42)
scaler_demo = StandardScaler()
X_demo_scaled = scaler_demo.fit_transform(X_demo)

db_demo = DBSCAN(eps=0.4, min_samples=5)
labels_demo = db_demo.fit_predict(X_demo_scaled)

core_mask = np.zeros(len(labels_demo), dtype=bool)
core_mask[db_demo.core_sample_indices_] = True
noise_mask = labels_demo == -1
border_mask = (~core_mask) & (~noise_mask)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

ax.scatter(X_demo_scaled[core_mask, 0], X_demo_scaled[core_mask, 1],
           c=labels_demo[core_mask], cmap='viridis', s=100,
           edgecolors='black', linewidth=1,
           label=f'Core Points ({core_mask.sum()})', zorder=3)

ax.scatter(X_demo_scaled[border_mask, 0], X_demo_scaled[border_mask, 1],
           c=labels_demo[border_mask], cmap='viridis', s=50,
           edgecolors='gray', linewidth=1, marker='s',
           label=f'Border Points ({border_mask.sum()})', zorder=2)

ax.scatter(X_demo_scaled[noise_mask, 0], X_demo_scaled[noise_mask, 1],
           c='red', s=80, marker='X', linewidth=1,
           label=f'Noise Points ({noise_mask.sum()})', zorder=4)

for idx in db_demo.core_sample_indices_[:2]:
    circle = plt.Circle((X_demo_scaled[idx, 0], X_demo_scaled[idx, 1]),
                         0.4, fill=False, color='blue', linewidth=1.5,
                         linestyle='--', alpha=0.5)
    ax.add_patch(circle)

ax.set_xlabel('Feature 0', fontsize=13); ax.set_ylabel('Feature 1', fontsize=13)
ax.set_title('DBSCAN Point Types (eps=0.4, min_samples=5)\n'
             'Blue dashed circles = eps neighborhood',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='upper right')
plt.tight_layout()
plt.savefig('plot_3_5_point_types.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                            ║
# ║  SECTION 4 — COMPARING & EVALUATING CLUSTERING (pp. 191–207)              ║
# ║                                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

## ── Section 4: Comparing & Evaluating Clustering ──

# ── 4.1 Adjusted Rand Index (ARI) — With Ground Truth ──
print("=" * 65)
print("4.1 Adjusted Rand Index (ARI)")
print("=" * 65)

X_eval, y_eval = make_moons(n_samples=200, noise=0.05, random_state=0)
scaler_eval = StandardScaler()
X_eval_scaled = scaler_eval.fit_transform(X_eval)

random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X_eval))

algorithms = {
    'Random': random_clusters,
    'k-Means': KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(X_eval_scaled),
    'Agglomerative': AgglomerativeClustering(n_clusters=2).fit_predict(X_eval_scaled),
    'DBSCAN': DBSCAN().fit_predict(X_eval_scaled),
}

fig, axes = plt.subplots(1, 4, figsize=(20, 4))

print('ARI Scores (1.0 = perfect, 0.0 = random):')
for ax, (name, labels) in zip(axes, algorithms.items()):
    ari = adjusted_rand_score(y_eval, labels)
    print(f'  {name:20s}: {ari:.2f}')
    ax.scatter(X_eval_scaled[:, 0], X_eval_scaled[:, 1], c=labels, cmap='viridis', s=40)
    ax.set_title(f'{name}\nARI: {ari:.2f}', fontsize=12)

plt.suptitle('Adjusted Rand Index (ARI) Comparison', fontsize=15, y=1.05)
plt.tight_layout()
plt.savefig('plot_4_1_ari.png', dpi=150, bbox_inches='tight')
plt.show()


# ── 4.2 Why accuracy_score is WRONG for Clustering ──
print("\n4.2 Why accuracy_score is WRONG for Clustering")

clusters1 = [0, 0, 1, 1, 0]
clusters2 = [1, 1, 0, 0, 1]  # Same grouping, labels just swapped!

print(f'Clusters1: {clusters1}')
print(f'Clusters2: {clusters2}  (identical grouping, different labels)')
print(f'\nAccuracy:  {accuracy_score(clusters1, clusters2):.2f}  <-- WRONG! Says 0%')
print(f'ARI:       {adjusted_rand_score(clusters1, clusters2):.2f}  <-- CORRECT! Says 100%')
print('Lesson: Cluster labels are arbitrary. Always use ARI or NMI, never accuracy!')


# ── 4.3 Silhouette Score — Without Ground Truth ──
print("\n4.3 Silhouette Score (No Ground Truth Needed)")

fig, axes = plt.subplots(1, 4, figsize=(20, 4))

print('Silhouette Scores (higher = more compact clusters):')
for ax, (name, labels) in zip(axes, algorithms.items()):
    n_unique = len(set(labels)) - (1 if -1 in labels else 0)
    sil = silhouette_score(X_eval_scaled, labels) if n_unique >= 2 else -1
    print(f'  {name:20s}: {sil:.2f}')
    ax.scatter(X_eval_scaled[:, 0], X_eval_scaled[:, 1], c=labels, cmap='viridis', s=40)
    ax.set_title(f'{name}\nSilhouette: {sil:.2f}', fontsize=12)

plt.suptitle('Silhouette Score Comparison', fontsize=15, y=1.05)
plt.tight_layout()
plt.savefig('plot_4_3_silhouette.png', dpi=150, bbox_inches='tight')
plt.show()
print('\nNote: k-Means scores HIGHER than DBSCAN even though DBSCAN is visually correct!')
print('Silhouette favors compact spherical clusters — it can be misleading.')


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                            ║
# ║  SECTION 5 — REAL-WORLD DEMO: IRIS DATASET (Enhancement)                  ║
# ║                                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

## ── Section 5: Real-World Demo — Iris Dataset ──

print("\n" + "=" * 65)
print("5. Real-World Demo — Iris Dataset")
print("=" * 65)

iris = load_iris()
X_iris = iris.data
y_iris = iris.target

print(f'Iris dataset: {X_iris.shape[0]} samples, {X_iris.shape[1]} features')
print(f'Features: {iris.feature_names}')
print(f'True classes: {iris.target_names}')

scaler_iris = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)
pca_iris = PCA(n_components=2)
X_iris_2d = pca_iris.fit_transform(X_iris_scaled)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

axes[0, 0].scatter(X_iris_2d[:, 0], X_iris_2d[:, 1], c=y_iris, cmap='viridis', s=50,
                    edgecolors='black', linewidth=0.5)
axes[0, 0].set_title('True Labels', fontsize=14)

labels_km_iris = KMeans(n_clusters=3, random_state=0, n_init=10).fit_predict(X_iris_scaled)
ari_km_iris = adjusted_rand_score(y_iris, labels_km_iris)
axes[0, 1].scatter(X_iris_2d[:, 0], X_iris_2d[:, 1], c=labels_km_iris, cmap='viridis', s=50,
                    edgecolors='black', linewidth=0.5)
axes[0, 1].set_title(f'k-Means (k=3) — ARI: {ari_km_iris:.2f}', fontsize=14)

labels_agg_iris = AgglomerativeClustering(n_clusters=3).fit_predict(X_iris_scaled)
ari_agg_iris = adjusted_rand_score(y_iris, labels_agg_iris)
axes[1, 0].scatter(X_iris_2d[:, 0], X_iris_2d[:, 1], c=labels_agg_iris, cmap='viridis', s=50,
                    edgecolors='black', linewidth=0.5)
axes[1, 0].set_title(f'Agglomerative — ARI: {ari_agg_iris:.2f}', fontsize=14)

labels_db_iris = DBSCAN(eps=0.9, min_samples=5).fit_predict(X_iris_scaled)
ari_db_iris = adjusted_rand_score(y_iris, labels_db_iris)
n_noise_iris = list(labels_db_iris).count(-1)
axes[1, 1].scatter(X_iris_2d[:, 0], X_iris_2d[:, 1], c=labels_db_iris, cmap='viridis', s=50,
                    edgecolors='black', linewidth=0.5)
axes[1, 1].set_title(f'DBSCAN — ARI: {ari_db_iris:.2f} ({n_noise_iris} noise pts)', fontsize=14)

for ax in axes.ravel():
    ax.set_xlabel('PCA Component 1'); ax.set_ylabel('PCA Component 2')

plt.suptitle('Clustering Algorithms on Iris Dataset', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig('plot_5_iris.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                            ║
# ║  SECTION 6 — BOOK EXAMPLE: LABELED FACES IN THE WILD (pp. 195–207)        ║
# ║                                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

## ── Section 6: Book's Real-World Example — Faces Dataset ──

print("\n" + "=" * 65)
print("6. Book Example — Labeled Faces in the Wild (pp. 195–207)")
print("=" * 65)

from sklearn.datasets import fetch_lfw_people

# ── 6.1 Load and Prepare ──
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

print(f'Dataset shape: {people.images.shape}')
print(f'Image size: {image_shape}')
print(f'Number of people: {len(people.target_names)}')

mask = np.zeros(people.target.shape, dtype=bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

print(f'Pixel value range: {X_people.min():.2f} to {X_people.max():.2f}')
if X_people.max() > 1.0:
    X_people = X_people / 255.
    print('Scaled pixels to 0-1 range')
else:
    print('Pixels already in 0-1 range')

print(f'After balancing: {X_people.shape[0]} images')


# ── 6.2 Sample Faces ──
fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})
fig.suptitle('Sample Faces from the Dataset', fontsize=16, fontweight='bold')
for i, (image, label, ax) in enumerate(zip(X_people, y_people, axes.ravel())):
    ax.imshow(image.reshape(image_shape), cmap='gray')
    ax.set_title(people.target_names[label].split()[-1], fontsize=11)
plt.tight_layout()
plt.savefig('plot_6_2_sample_faces.png', dpi=150, bbox_inches='tight')
plt.show()


# ── 6.3 PCA Preprocessing ──
pca_faces = PCA(n_components=100, whiten=True, random_state=0)
pca_faces.fit(X_people)
X_pca = pca_faces.transform(X_people)

print(f'\nPCA: {X_people.shape} → {X_pca.shape}')
print(f'Variance explained: {pca_faces.explained_variance_ratio_.sum():.1%}')


# ── 6.4 DBSCAN on Faces — Outlier Detection (pp. 195–199) ──
print('\n--- DBSCAN Tuning Process ---')

print(f'Default (eps=0.5): All noise — eps too small for 100D data')
print(f'min_samples=3 (eps=0.5): Still all noise — eps is the problem')

print('\n--- Exploring different eps values ---')
for eps in [1, 3, 5, 7, 9, 11, 13, 15]:
    dbscan = DBSCAN(eps=eps, min_samples=3)
    labels = dbscan.fit_predict(X_pca)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    sizes = np.bincount(labels + 1)
    print(f'  eps={eps:>2}: {n_clusters} clusters, {n_noise} noise | sizes: {sizes}')


# ── 6.5 Show Outlier Faces (eps=15) ──
dbscan_15 = DBSCAN(min_samples=3, eps=15)
labels_15 = dbscan_15.fit_predict(X_pca)

noise_mask_faces = labels_15 == -1
n_noise_faces = noise_mask_faces.sum()
print(f'\neps=15: {n_noise_faces} noise points (outlier faces)')

noise_images = X_people[noise_mask_faces]
n_show = min(n_noise_faces, 15)
cols = min(n_show, 5)
rows = max((n_show + cols - 1) // cols, 1)

fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows),
                         subplot_kw={'xticks': (), 'yticks': ()})
fig.suptitle('DBSCAN Noise Points — Outlier Faces (eps=15)',
             fontsize=15, fontweight='bold', color='red')

axes_flat = axes.ravel() if hasattr(axes, 'ravel') else [axes]
for i, ax in enumerate(axes_flat):
    if i < n_show:
        ax.imshow(noise_images[i].reshape(image_shape), cmap='gray')
        ax.set_title(f'Outlier {i+1}', fontsize=10, color='red')
    else:
        ax.set_visible(False)

plt.tight_layout()
plt.savefig('plot_6_5_outlier_faces.png', dpi=150, bbox_inches='tight')
plt.show()
print('OUTLIER DETECTION — a unique strength of DBSCAN!')


# ── 6.6 DBSCAN eps=7 — Small Similar Clusters ──
dbscan_7 = DBSCAN(min_samples=3, eps=7)
labels_7 = dbscan_7.fit_predict(X_pca)

n_clusters_7 = max(labels_7) + 1
n_noise_7 = list(labels_7).count(-1)
print(f'\neps=7: {n_clusters_7} clusters, {n_noise_7} noise points')

for cluster in range(min(n_clusters_7, 10)):
    mask_c = labels_7 == cluster
    n_images = np.sum(mask_c)
    if n_images == 0:
        continue

    n_show = min(n_images, 8)
    fig, axes = plt.subplots(1, n_show + 1, figsize=(2.5 * (n_show + 1), 3),
                             subplot_kw={'xticks': (), 'yticks': ()})

    axes[0].text(0.5, 0.5, f'Cluster {cluster}\n({n_images} faces)',
                 ha='center', va='center', fontsize=12, fontweight='bold',
                 transform=axes[0].transAxes)
    axes[0].set_frame_on(False)

    images_c = X_people[mask_c]
    labels_true_c = y_people[mask_c]
    for i, ax in enumerate(axes[1:]):
        if i < n_show:
            ax.imshow(images_c[i].reshape(image_shape), cmap='gray')
            ax.set_title(people.target_names[labels_true_c[i]].split()[-1], fontsize=9)

    plt.tight_layout()
    plt.show()

print('Each cluster contains genuinely similar faces.')


# ── 6.7 k-Means on Faces — Average Faces (pp. 200–202) ──
print('\n--- k-Means on Faces ---')

km_faces = KMeans(n_clusters=10, random_state=0, n_init=10)
labels_km_faces = km_faces.fit_predict(X_pca)

print(f'k-Means cluster sizes: {np.bincount(labels_km_faces)}')

fig, axes = plt.subplots(2, 5, figsize=(15, 7),
                         subplot_kw={'xticks': (), 'yticks': ()})
fig.suptitle('k-Means Cluster Centers — "Average Faces" (k=10)',
             fontsize=15, fontweight='bold')

for i, (center, ax) in enumerate(zip(km_faces.cluster_centers_, axes.ravel())):
    face = pca_faces.inverse_transform(center)
    ax.imshow(face.reshape(image_shape), cmap='gray')
    size = np.sum(labels_km_faces == i)
    ax.set_title(f'Cluster {i} ({size} faces)', fontsize=11)

plt.tight_layout()
plt.savefig('plot_6_7_kmeans_faces.png', dpi=150, bbox_inches='tight')
plt.show()


# ── 6.8 Agglomerative on Faces + Dendrogram (pp. 203–207) ──
print('\n--- Agglomerative on Faces ---')

agg_faces = AgglomerativeClustering(n_clusters=10)
labels_agg_faces = agg_faces.fit_predict(X_pca)

print(f'Agglomerative cluster sizes: {np.bincount(labels_agg_faces)}')
print(f'ARI between k-Means and Agglomerative: {adjusted_rand_score(labels_km_faces, labels_agg_faces):.2f}')

linkage_array = ward(X_pca)

plt.figure(figsize=(20, 6))
dendrogram(linkage_array, p=7, truncate_mode='level', no_labels=True)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Cluster Distance', fontsize=12)
plt.title('Dendrogram of Faces Dataset — No clear natural number of clusters',
          fontsize=15, fontweight='bold')
plt.savefig('plot_6_8_faces_dendrogram.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                            ║
# ║  ALGORITHM COMPARISON TABLE                                                ║
# ║                                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print('\n' + '=' * 75)
print('ALGORITHM COMPARISON — STRENGTHS & WEAKNESSES')
print('=' * 75)
print(f"""
┌─────────────────────┬──────────────┬────────────────┬──────────────┐
│ Feature             │ k-Means      │ Agglomerative  │ DBSCAN       │
├─────────────────────┼──────────────┼────────────────┼──────────────┤
│ Must set # clusters │ Yes          │ Yes            │ No  ✓        │
│ Complex shapes      │ No           │ No             │ Yes ✓        │
│ Noise detection     │ No           │ No             │ Yes ✓        │
│ Scalability         │ Excellent ✓  │ Good           │ Good         │
│ predict() new data  │ Yes ✓        │ No             │ No           │
│ Varying densities   │ Partial      │ Partial        │ Struggles    │
│ Cluster shapes      │ Spherical    │ Spherical      │ Arbitrary ✓  │
│ Key output          │ Centers      │ Dendrogram     │ Core/Border  │
│ Parameters          │ n_clusters   │ n_clusters,    │ eps,         │
│                     │              │ linkage        │ min_samples  │
└─────────────────────┴──────────────┴────────────────┴──────────────┘
""")

print("=" * 75)
print("END OF NOTEBOOK")
print("=" * 75)
