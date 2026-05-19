# =========================================================
# ALGORITMOS NO SUPERVISADOS SOBRE UN DATASET CSV
# Algoritmos:
# - Agglomerative Clustering
# - DBSCAN
# - HDBSCAN
# - OPTICS
# - t-SNE
# - PCA
# - KMeans
# =========================================================

# INSTALAR:
# pip install pandas matplotlib scikit-learn hdbscan seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# Clustering
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    OPTICS,
    KMeans
)

import hdbscan

# Reducción de dimensionalidad
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# =========================================================
# 1. CARGAR DATASET
# =========================================================

# Cambia por el nombre de tu archivo
dataset = "dataset.csv"

df = pd.read_csv(dataset)

print("Primeras filas:")
print(df.head())

print("\nInformación:")
print(df.info())

# =========================================================
# 2. LIMPIEZA BÁSICA
# =========================================================

# Seleccionar solo columnas numéricas
X = df.select_dtypes(include=[np.number])

# Eliminar filas con valores nulos
X = X.dropna()

print("\nDataset numérico:")
print(X.head())

# =========================================================
# 3. NORMALIZACIÓN
# =========================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================================================
# 4. PCA (2 COMPONENTES)
# =========================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1])

plt.title("PCA")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.grid()

plt.show()

# =========================================================
# 5. t-SNE
# =========================================================

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1])

plt.title("t-SNE")
plt.xlabel("Dimensión 1")
plt.ylabel("Dimensión 2")
plt.grid()

plt.show()

# =========================================================
# 6. KMEANS
# =========================================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

kmeans_labels = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=kmeans_labels
)

plt.title("KMeans")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid()

plt.show()

# =========================================================
# 7. AGGLOMERATIVE CLUSTERING
# =========================================================

agg = AgglomerativeClustering(
    n_clusters=3
)

agg_labels = agg.fit_predict(X_scaled)

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=agg_labels
)

plt.title("Agglomerative Clustering")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid()

plt.show()

# =========================================================
# 8. DBSCAN
# =========================================================

dbscan = DBSCAN(
    eps=0.8,
    min_samples=5
)

dbscan_labels = dbscan.fit_predict(X_scaled)

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=dbscan_labels
)

plt.title("DBSCAN")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid()

plt.show()

# =========================================================
# 9. HDBSCAN
# =========================================================

hdb = hdbscan.HDBSCAN(
    min_cluster_size=10
)

hdb_labels = hdb.fit_predict(X_scaled)

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=hdb_labels
)

plt.title("HDBSCAN")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid()

plt.show()

# =========================================================
# 10. OPTICS
# =========================================================

optics = OPTICS(
    min_samples=5
)

optics_labels = optics.fit_predict(X_scaled)

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=optics_labels
)

plt.title("OPTICS")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid()

plt.show()

# =========================================================
# 11. RESUMEN DE CLUSTERS
# =========================================================

resultado = pd.DataFrame({
    "KMeans": kmeans_labels,
    "Agglomerative": agg_labels,
    "DBSCAN": dbscan_labels,
    "HDBSCAN": hdb_labels,
    "OPTICS": optics_labels
})

print("\nResultado de clusters:")
print(resultado.head())

# =========================================================
# FIN
# =========================================================