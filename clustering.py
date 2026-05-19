import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS

# 1. CARGA Y PREPROCESAMIENTO DEL DATASET
# =====================================================================
try:
    # Reemplaza 'dataset.csv' con la ruta real de tu archivo
    df = pd.read_csv('extraccion_dataset.csv') 
    print("¡Dataset cargado con éxito!")
except FileNotFoundError:
    print("Error: No se encontró el archivo 'dataset.csv'. Creando datos sintéticos para la prueba...")
    # Creamos datos de prueba que simulan tu estructura (Columnas 0,1,2 texto/otros, columna 3 'Titulo')
    from sklearn.datasets import make_blobs
    X_synthetic, _ = make_blobs(n_samples=300, n_features=6, centers=3, random_state=42)
    df = pd.DataFrame(X_synthetic, columns=[f'num_col_{i}' for i in range(6)])
    df.insert(0, 'Col_Inutil_1', 'A')
    df.insert(1, 'Col_Inutil_2', 'B')
    df.insert(2, 'Col_Inutil_3', 'C')
    # Añadimos la columna Titulo con categorías ficticias para etiquetar
    df.insert(3, 'Titulo', np.random.choice(['Categoría A', 'Categoría B', 'Categoría C'], size=300))

# --- Extracción de datos según tus especificaciones ---
# Guardamos la columna 'Titulo' para el etiquetado
labels_titulo = df['Titulo']

# Seleccionamos las columnas numéricas a partir de la 4ta columna (Índice 3 en adelante)
X_numeric = df.iloc[:, 3:].select_dtypes(include=[np.number])
idx_validos = X_numeric.dropna().index
X_numeric = X_numeric.loc[idx_validos]
labels_titulo = labels_titulo.loc[idx_validos]

print(f"Columnas numéricas detectadas y utilizadas: {list(X_numeric.columns)}")

# Escalamiento de características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Configuración de la matriz de gráficos (3 filas x 2 columnas = 6 gráficos)
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
axes = axes.flatten()
plot_index = 0

# =====================================================================
# FASE 1: REDUCCIÓN DE DIMENSIONALIDAD (Etiquetado con la columna 'Titulo')
# =====================================================================

# --- 1. PCA (Principal Component Analysis) ---
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Usamos 'Titulo' como hue para ver la distribución real de tus etiquetas
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_titulo, ax=axes[plot_index], palette='Set2', alpha=0.8)
axes[plot_index].set_title('1. PCA (Color por "Titulo")')
axes[plot_index].set_xlabel('Componente Principal 1')
axes[plot_index].set_ylabel('Componente Principal 2')
axes[plot_index].legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Titulo")
plot_index += 1

# --- 2. t-SNE ---
tsne = TSNE(n_components=2, perplexity=10, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels_titulo, ax=axes[plot_index], palette='Set2', alpha=0.8)
axes[plot_index].set_title('2. t-SNE (Color por "Titulo")')
axes[plot_index].set_xlabel('Dimensión t-SNE 1')
axes[plot_index].set_ylabel('Dimensión t-SNE 2')
axes[plot_index].legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Titulo")
plot_index += 1


# =====================================================================
# FASE 2: ALGORITMOS DE CLUSTERING NO SUPERVISADO (Visualizados sobre PCA)
# =====================================================================

# --- 3. K-Means ---
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
labels_kmeans = kmeans.fit_predict(X_pca)

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_kmeans, ax=axes[plot_index], palette='tab10', alpha=0.7)
axes[plot_index].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=150, c='red', marker='X', label='Centroides')
axes[plot_index].set_title('3. K-Means Clustering')
axes[plot_index].legend(title="Cluster")
plot_index += 1

# --- 4. Agglomerative Clustering ---
agg = AgglomerativeClustering(n_clusters=3)
labels_agg = agg.fit_predict(X_pca)

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_agg, ax=axes[plot_index], palette='tab10', alpha=0.7)
axes[plot_index].set_title('4. Agglomerative Clustering')
axes[plot_index].legend(title="Cluster")
plot_index += 1

# --- 5. DBSCAN ---
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_pca)

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_dbscan, ax=axes[plot_index], palette='deep', alpha=0.7)
axes[plot_index].set_title('5. DBSCAN (Etiqueta -1 es Ruido)')
axes[plot_index].legend(title="Cluster")
plot_index += 1

# --- 6. OPTICS ---
optics = OPTICS(min_samples=15, xi=0.05, min_cluster_size=0.1)
labels_optics = optics.fit_predict(X_pca)

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_optics, ax=axes[plot_index], palette='deep', alpha=0.7)
axes[plot_index].set_title('6. OPTICS Clustering')
axes[plot_index].legend(title="Cluster")
plot_index += 1

# Ajustar diseño final para evitar superposiciones de leyendas
plt.tight_layout()
plt.show()