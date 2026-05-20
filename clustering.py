import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS

try:
    df = pd.read_csv('extraccion_dataset.csv') 
    print("¡Dataset cargado con éxito!")
except FileNotFoundError:
    from sklearn.datasets import make_blobs
    X_synthetic, _ = make_blobs(n_samples=300, n_features=6, centers=3, random_state=42)
    df = pd.DataFrame(X_synthetic, columns=[f'num_col_{i}' for i in range(6)])
    df.insert(0, 'Col_Inutil_1', 'A')
    df.insert(1, 'Col_Inutil_2', 'B')
    df.insert(2, 'Col_Inutil_3', 'C')
    df.insert(3, 'Titulo', np.random.choice(['Categoría A', 'Categoría B', 'Categoría C'], size=300))

labels_titulo = df['Titulo']
X_numeric = df.iloc[:, 3:].select_dtypes(include=[np.number])
idx_validos = X_numeric.dropna().index
X_numeric = X_numeric.loc[idx_validos]
labels_titulo = labels_titulo.loc[idx_validos]

print(f"Columnas numéricas detectadas: {list(X_numeric.columns)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# pca
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# tsne
tsne = TSNE(n_components=2, perplexity=10, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# kmeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
labels_kmeans = kmeans.fit_predict(X_pca)

# agglomerative
agg = AgglomerativeClustering(n_clusters=3)
labels_agg = agg.fit_predict(X_pca)

# dbscan
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_pca)

# optics
optics = OPTICS(min_samples=15, xi=0.05, min_cluster_size=0.1)
labels_optics = optics.fit_predict(X_pca)

def add_titulo_labels(x, y, labels, ax, font_size=8, sample_step=1):
    for i in range(0, len(x), sample_step):
        ax.annotate(labels.iloc[i], 
                   (x[i], y[i]), 
                   fontsize=font_size, 
                   alpha=0.6,
                   xytext=(5, 5), 
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

# pca
plt.figure(figsize=(12, 9))
ax = plt.gca()
scatter = sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_titulo, 
                          palette='Set2', alpha=0.8, s=100, edgecolor='black', linewidth=0.5)
add_titulo_labels(X_pca[:, 0], X_pca[:, 1], labels_titulo, ax, font_size=7, sample_step=10)
plt.title('1. PCA (Color por "Titulo")', fontsize=14, fontweight='bold')
plt.xlabel('Componente Principal 1', fontsize=12)
plt.ylabel('Componente Principal 2', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Titulo", title_fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# tsne
plt.figure(figsize=(12, 9))
ax = plt.gca()
scatter = sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels_titulo, 
                          palette='Set2', alpha=0.8, s=100, edgecolor='black', linewidth=0.5)
add_titulo_labels(X_tsne[:, 0], X_tsne[:, 1], labels_titulo, ax, font_size=7, sample_step=10)
plt.title('2. t-SNE (Color por "Titulo")', fontsize=14, fontweight='bold')
plt.xlabel('Dimensión t-SNE 1', fontsize=12)
plt.ylabel('Dimensión t-SNE 2', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Titulo", title_fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# kmeans
plt.figure(figsize=(12, 9))
ax = plt.gca()
scatter = sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_kmeans, 
                          palette='tab10', alpha=0.7, s=100, edgecolor='black', linewidth=0.5)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           s=300, c='red', marker='X', label='Centroides', edgecolor='black', linewidth=2)
add_titulo_labels(X_pca[:, 0], X_pca[:, 1], labels_titulo, ax, font_size=7, sample_step=10)
plt.title('3. K-Means Clustering', fontsize=14, fontweight='bold')
plt.xlabel('Componente Principal 1', fontsize=12)
plt.ylabel('Componente Principal 2', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Cluster", title_fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# agglomerative
plt.figure(figsize=(12, 9))
ax = plt.gca()
scatter = sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_agg, 
                          palette='tab10', alpha=0.7, s=100, edgecolor='black', linewidth=0.5)
add_titulo_labels(X_pca[:, 0], X_pca[:, 1], labels_titulo, ax, font_size=7, sample_step=10)
plt.title('4. Agglomerative Clustering', fontsize=14, fontweight='bold')
plt.xlabel('Componente Principal 1', fontsize=12)
plt.ylabel('Componente Principal 2', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Cluster", title_fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# dbscan
plt.figure(figsize=(12, 9))
ax = plt.gca()
scatter = sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_dbscan, 
                          palette='deep', alpha=0.7, s=100, edgecolor='black', linewidth=0.5)
add_titulo_labels(X_pca[:, 0], X_pca[:, 1], labels_titulo, ax, font_size=7, sample_step=10)
plt.title('5. DBSCAN (Etiqueta -1 es Ruido)', fontsize=14, fontweight='bold')
plt.xlabel('Componente Principal 1', fontsize=12)
plt.ylabel('Componente Principal 2', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Cluster", title_fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# optics
plt.figure(figsize=(12, 9))
ax = plt.gca()
scatter = sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_optics, 
                          palette='deep', alpha=0.7, s=100, edgecolor='black', linewidth=0.5)
add_titulo_labels(X_pca[:, 0], X_pca[:, 1], labels_titulo, ax, font_size=7, sample_step=10)
plt.title('6. OPTICS Clustering', fontsize=14, fontweight='bold')
plt.xlabel('Componente Principal 1', fontsize=12)
plt.ylabel('Componente Principal 2', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Cluster", title_fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()