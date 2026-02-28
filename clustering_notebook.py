"""
Unsupervised Learning — Clustering Algorithms
Course: CSC564 — Machine Learning | King Saud University
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
from sklearn.metrics import accuracy_score, pairwise_distances_argmin
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, ward

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print("All imports successful!\n")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                            ║
# ║  SECTION 1 — K-MEANS CLUSTERING (pp. 168–181)                             ║
# ║                                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

## ── Section 1: K-Means Clustering ──

# ── 1.1 Manual k-Means with Decision Boundaries ──
print("=" * 65)
print("1.1 Manual k-Means with Decision Boundaries")
print("=" * 65)

# Generate Synthetic Data (300 samples, 4 natural clusters)
X_km, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

k_value = 4  # number of clusters

# Initialize: pick random points as starting centers
rng = np.random.RandomState(42)
initial_indices = rng.permutation(X_km.shape[0])[:k_value]
centers = X_km[initial_indices]


# Function to draw the decision boundaries (Voronoi regions)
def plot_decision_boundaries(ax, centers, X):
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Calculate Euclidean distance for every point in the background grid
    Z = pairwise_distances_argmin(np.c_[xx.ravel(), yy.ravel()], centers)
    Z = Z.reshape(xx.shape)

    # Plot the color-coded regions
    ax.imshow(Z, interpolation='nearest',
              extent=(xx.min(), xx.max(), yy.min(), yy.max()),
              cmap='Pastel1', aspect='auto', origin='lower', alpha=0.4)


# Run k-Means manually, step by step
iteration = 0
history = []  # store (centers, labels) at each step

while True:
    iteration += 1
    # Assignment Step: assign each point to nearest center (Euclidean distance)
    labels = pairwise_distances_argmin(X_km, centers)
    history.append((centers.copy(), labels.copy()))

    # Update Step: move centers to the mean of their assigned points
    new_centers = np.array([X_km[labels == j].mean(0) for j in range(k_value)])

    # Check for convergence
    if np.all(centers == new_centers):
        print(f"Converged after {iteration} iterations!")
        break
    centers = new_centers

# Show iterations: first, middle, and final
show_iters = [0, len(history) // 2, len(history) - 1]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, idx in zip(axes, show_iters):
    c, l = history[idx]
    plot_decision_boundaries(ax, c, X_km)
    ax.scatter(X_km[:, 0], X_km[:, 1], c=l, s=40, cmap='viridis', edgecolors='k', linewidth=0.3)
    ax.scatter(c[:, 0], c[:, 1], c='red', s=250, marker='^', edgecolors='black',
               linewidth=1.5, label='Centers', zorder=5)
    ax.set_title(f'Iteration {idx + 1}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
    ax.legend(fontsize=10)

plt.suptitle(f'Manual k-Means Step-by-Step (k={k_value}, converged in {len(history)} iterations)',
             fontsize=16, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('plot_1_1_kmeans_manual.png', dpi=150, bbox_inches='tight')
plt.show()

# Final result with full decision boundaries
fig, ax = plt.subplots(figsize=(10, 8))
final_centers, final_labels = history[-1]
plot_decision_boundaries(ax, final_centers, X_km)
ax.scatter(X_km[:, 0], X_km[:, 1], c=final_labels, s=50, cmap='viridis', edgecolors='k')
ax.scatter(final_centers[:, 0], final_centers[:, 1], c='red', s=250, marker='^',
           edgecolors='black', linewidth=1.5, label='Cluster Centers')
ax.set_title(f'Final Decision Boundaries (k={k_value})\nTotal Iterations: {len(history)}',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Feature 0')
ax.set_ylabel('Feature 1')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig('plot_1_1_kmeans_final.png', dpi=150, bbox_inches='tight')
plt.show()


# ── 1.2 k-Means Failure Cases ──
print("\n1.2 k-Means Failure Cases")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Failure 1: Different densities
X_dense, y_dense = make_blobs(n_samples=300, cluster_std=[1.0, 2.5, 0.5], random_state=170)
labels_dense = KMeans(n_clusters=3, random_state=0, n_init=10).fit_predict(X_dense)
axes[0].scatter(X_dense[:, 0], X_dense[:, 1], c=labels_dense, cmap='viridis', s=40, edgecolors='k', linewidth=0.3)
axes[0].set_title('Different Densities\n(misassigns sparse cluster)', fontsize=13, fontweight='bold')

# Failure 2: Elongated/non-spherical
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(make_blobs(n_samples=300, random_state=170)[0], transformation)
labels_aniso = KMeans(n_clusters=3, random_state=0, n_init=10).fit_predict(X_aniso)
axes[1].scatter(X_aniso[:, 0], X_aniso[:, 1], c=labels_aniso, cmap='viridis', s=40, edgecolors='k', linewidth=0.3)
axes[1].set_title('Elongated Clusters\n(cuts through natural groups)', fontsize=13, fontweight='bold')

# Failure 3: Two moons
X_moons_fail, y_moons_fail = make_moons(n_samples=200, noise=0.05, random_state=0)
labels_moons_fail = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(X_moons_fail)
axes[2].scatter(X_moons_fail[:, 0], X_moons_fail[:, 1], c=labels_moons_fail, cmap='viridis', s=40, edgecolors='k', linewidth=0.3)
axes[2].set_title('Two Moons\n(spherical assumption fails)', fontsize=13, fontweight='bold')

for ax in axes:
    ax.set_xlabel('Feature 0'); ax.set_ylabel('Feature 1')

plt.suptitle('k-Means Failure Cases — When Spherical Assumption Breaks',
             fontsize=16, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('plot_1_2_kmeans_failures.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                            ║
# ║  SECTION 2 — AGGLOMERATIVE CLUSTERING (pp. 182–187)                       ║
# ║                                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

## ── Section 2: Agglomerative Clustering ──

# ── 2.1 Basic Agglomerative Clustering ──
print("\n" + "=" * 65)
print("2.1 Basic Agglomerative Clustering")
print("=" * 65)

X_agg, y_agg = make_blobs(random_state=1)

agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_agg = agg.fit_predict(X_agg)

plt.figure(figsize=(6, 5))
plt.scatter(X_agg[:, 0], X_agg[:, 1], c=labels_agg, s=100, cmap='viridis',
            edgecolors='k', linewidth=0.5)

plt.title("Agglomerative Clustering", fontsize=14, fontweight='bold')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.savefig('plot_2_1_agglomerative.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'Cluster labels: {labels_agg}')


# ── 2.2 Dendrogram ──
print("\n2.2 Dendrogram (Ward Linkage)")

linked = linkage(X_agg, method='ward')

plt.figure(figsize=(9, 6))
dendrogram(linked)
plt.title("Dendrogram with (Ward)", fontsize=14, fontweight='bold')
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig('plot_2_2_dendrogram.png', dpi=150, bbox_inches='tight')
plt.show()


# ── 2.3 Agglomerative on Larger Dataset + Linkage Comparison ──
print("\n2.3 Linkage Comparison (Ward vs Complete vs Average)")

# Use ELONGATED (anisotropic) blobs — linkage choice matters here!
# With well-separated round blobs all linkages give identical results.
X_aniso_raw, _ = make_blobs(n_samples=150, centers=3, random_state=170)
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X_aniso_raw, transformation)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
linkage_types = ['ward', 'complete', 'average']

for ax, link_type in zip(axes, linkage_types):
    agg = AgglomerativeClustering(n_clusters=3, linkage=link_type)
    labels = agg.fit_predict(X_aniso)
    ax.scatter(X_aniso[:, 0], X_aniso[:, 1], c=labels, s=50, cmap='viridis',
               edgecolors='k', linewidth=0.3)
    ax.set_title(f'Linkage: {link_type.capitalize()}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.suptitle('Agglomerative Clustering — Linkage Comparison on Elongated Data (k=3)\n'
             'Notice: each linkage draws different boundaries!',
             fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('plot_2_3_linkage_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                            ║
# ║  SECTION 3 — DBSCAN CLUSTERING (pp. 187–190)                              ║
# ║  Density-Based Spatial Clustering of Applications with Noise               ║
# ║                                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

## ── Section 3: DBSCAN Clustering ──

# ── 3.1 DBSCAN Parameter Exploration (Table) ──
print("\n" + "=" * 65)
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

labels_agg_m = AgglomerativeClustering(n_clusters=2).fit_predict(X_moons2_scaled)
axes[1].scatter(X_moons2_scaled[:, 0], X_moons2_scaled[:, 1], c=labels_agg_m, cmap='viridis', s=60)
axes[1].set_title('Agglomerative (k=2)\n❌ Also fails!', fontsize=13)

labels_db_m = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_moons2_scaled)
axes[2].scatter(X_moons2_scaled[:, 0], X_moons2_scaled[:, 1], c=labels_db_m, cmap='viridis', s=60)
axes[2].set_title('DBSCAN (eps=0.5)\n✅ Correctly separates!', fontsize=13)

for ax in axes:
    ax.set_xlabel('Feature 0'); ax.set_ylabel('Feature 1')

plt.suptitle('Two Moons: DBSCAN Succeeds Where Others Fail', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('plot_3_4_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# ── 3.5 DBSCAN Core / Border / Noise Visualization ──
print("3.5 DBSCAN Core / Border / Noise Visualization")

# Use eps=0.3 and noise=0.1 to get a clear mix of core, border, AND noise points.
# (The old version with eps=0.4/noise=0.12 had 0 noise — defeating the purpose!)
from sklearn.metrics import pairwise_distances as pdist_full

X_demo, _ = make_moons(n_samples=200, noise=0.1, random_state=42)
X_demo_scaled = StandardScaler().fit_transform(X_demo)

EPS_DEMO = 0.3
MIN_S_DEMO = 5
db_demo = DBSCAN(eps=EPS_DEMO, min_samples=MIN_S_DEMO)
labels_demo = db_demo.fit_predict(X_demo_scaled)

core_mask = np.zeros(len(labels_demo), dtype=bool)
core_mask[db_demo.core_sample_indices_] = True
noise_mask = labels_demo == -1
border_mask = (~core_mask) & (~noise_mask)

# Compute pairwise distances for neighbor counting
dists_demo = pdist_full(X_demo_scaled)

# Pick annotated examples from DIFFERENT regions of the plot
# Core: dense upper area
best_core = db_demo.core_sample_indices_[0]
for idx in db_demo.core_sample_indices_:
    x, y = X_demo_scaled[idx]
    if -0.8 < x < -0.2 and 1.0 < y < 1.5 and np.sum(dists_demo[idx] <= EPS_DEMO) >= 6:
        best_core = idx; break

# Border: bottom-left edge
best_border = np.where(border_mask)[0][0]
for idx in np.where(border_mask)[0]:
    x, y = X_demo_scaled[idx]
    if x < -1.3 and y < 0:
        best_border = idx; break

# Noise: most isolated point on the right
best_noise, best_iso = np.where(noise_mask)[0][0], 0
for idx in np.where(noise_mask)[0]:
    x, y = X_demo_scaled[idx]
    if x > 0.5:
        min_d = np.sort(dists_demo[idx])[1]
        if min_d > best_iso:
            best_iso = min_d; best_noise = idx

core_n = np.sum(dists_demo[best_core] <= EPS_DEMO)
border_n = np.sum(dists_demo[best_border] <= EPS_DEMO)
noise_n = np.sum(dists_demo[best_noise] <= EPS_DEMO)
print(f'  Core example:   {core_n} pts in eps (>= {MIN_S_DEMO} → core)')
print(f'  Border example: {border_n} pts in eps (< {MIN_S_DEMO}, but near a core → border)')
print(f'  Noise example:  {noise_n} pts in eps (< {MIN_S_DEMO}, no core nearby → noise)')

# ── PLOT ──
# KEY FIX: color by POINT TYPE (core/border/noise), NOT by cluster label.
# The old version colored by cluster, which confused the yellow core point —
# it looked like it had only 3 yellow neighbors, but neighbors of ANY cluster count!
fig, ax = plt.subplots(figsize=(14, 9))

ax.scatter(X_demo_scaled[core_mask, 0], X_demo_scaled[core_mask, 1],
           c='#2166AC', s=70, edgecolors='black', linewidth=0.4,
           label=f'Core ({core_mask.sum()}) — >= {MIN_S_DEMO} pts within eps', zorder=3, alpha=0.75)
ax.scatter(X_demo_scaled[border_mask, 0], X_demo_scaled[border_mask, 1],
           c='#FDAE61', s=110, edgecolors='black', linewidth=1, marker='s',
           label=f'Border ({border_mask.sum()}) — < {MIN_S_DEMO} pts, but near a core point', zorder=4)
ax.scatter(X_demo_scaled[noise_mask, 0], X_demo_scaled[noise_mask, 1],
           c='#D73027', s=130, edgecolors='black', linewidth=1.5, marker='X',
           label=f'Noise ({noise_mask.sum()}) — < {MIN_S_DEMO} pts AND not near any core', zorder=5)

# Annotated CORE example
cx, cy = X_demo_scaled[best_core]
ax.add_patch(plt.Circle((cx, cy), EPS_DEMO, fill=True, facecolor='#2166AC', alpha=0.1,
             edgecolor='#2166AC', linewidth=2.5, linestyle='--', zorder=2))
ax.plot(cx, cy, 'o', color='#2166AC', markersize=14, markeredgecolor='white', markeredgewidth=2.5, zorder=8)
ax.annotate(f'CORE POINT\n{core_n} pts within eps (>= {MIN_S_DEMO}) ✓\nStarts or joins a cluster',
            xy=(cx, cy), xytext=(cx + 1.0, cy + 0.5), fontsize=11, fontweight='bold', color='#2166AC',
            arrowprops=dict(arrowstyle='->', color='#2166AC', lw=2.5),
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#EBF0FA', edgecolor='#2166AC', linewidth=2), zorder=10)

# Annotated BORDER example
bx, by = X_demo_scaled[best_border]
ax.add_patch(plt.Circle((bx, by), EPS_DEMO, fill=True, facecolor='#FDAE61', alpha=0.12,
             edgecolor='#E08214', linewidth=2.5, linestyle='--', zorder=2))
ax.plot(bx, by, 's', color='#FDAE61', markersize=14, markeredgecolor='black', markeredgewidth=2, zorder=8)
ax.annotate(f'BORDER POINT\n{border_n} pts within eps (< {MIN_S_DEMO}) ✗\nBut within eps of a core point',
            xy=(bx, by), xytext=(bx + 0.5, by - 0.9), fontsize=11, fontweight='bold', color='#CC6600',
            arrowprops=dict(arrowstyle='->', color='#CC6600', lw=2.5),
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF5E6', edgecolor='#CC6600', linewidth=2), zorder=10)

# Annotated NOISE example
nx, ny = X_demo_scaled[best_noise]
ax.add_patch(plt.Circle((nx, ny), EPS_DEMO, fill=True, facecolor='#D73027', alpha=0.08,
             edgecolor='#D73027', linewidth=2.5, linestyle='--', zorder=2))
ax.plot(nx, ny, 'X', color='#D73027', markersize=16, markeredgecolor='black', markeredgewidth=1.5, zorder=8)
ax.annotate(f'NOISE POINT\n{noise_n} pt within eps (< {MIN_S_DEMO}) ✗\nNo core point nearby either',
            xy=(nx, ny), xytext=(nx - 1.5, ny + 0.8), fontsize=11, fontweight='bold', color='#D73027',
            arrowprops=dict(arrowstyle='->', color='#D73027', lw=2.5),
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEEEE', edgecolor='#D73027', linewidth=2), zorder=10)

ax.set_xlabel('Feature 0', fontsize=14); ax.set_ylabel('Feature 1', fontsize=14)
ax.set_title(f'DBSCAN Point Types (eps={EPS_DEMO}, min_samples={MIN_S_DEMO})\n'
             f'Dashed circles = eps neighborhood — count ALL points inside',
             fontsize=15, fontweight='bold')
ax.legend(fontsize=12, loc='upper left', framealpha=0.95, edgecolor='gray')
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
print("\n" + "=" * 65)
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
