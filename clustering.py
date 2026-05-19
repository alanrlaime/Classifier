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

# =====================================================================
# FASE 1: REDUCCIÓN DE DIMENSIONALIDAD (Etiquetado con la columna 'Titulo')
# =====================================================================

print("\n--- Calculando PCA ---")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("--- Calculando t-SNE ---")
tsne = TSNE(n_components=2, perplexity=10, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# =====================================================================
# FASE 2: ALGORITMOS DE CLUSTERING NO SUPERVISADO (Visualizados sobre PCA)
# =====================================================================

print("--- Calculando K-Means ---")
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
labels_kmeans = kmeans.fit_predict(X_pca)

print("--- Calculando Agglomerative Clustering ---")
agg = AgglomerativeClustering(n_clusters=3)
labels_agg = agg.fit_predict(X_pca)

print("--- Calculando DBSCAN ---")
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_pca)

print("--- Calculando OPTICS ---")
optics = OPTICS(min_samples=15, xi=0.05, min_cluster_size=0.1)
labels_optics = optics.fit_predict(X_pca)

# =====================================================================
# FUNCIÓN PARA AGREGAR ETIQUETAS DE TÍTULO A CADA PUNTO
# =====================================================================
def add_titulo_labels(x, y, labels, ax, font_size=8, sample_step=1):
    """
    Agrega las etiquetas de título a cada punto del gráfico
    
    Parameters:
    - x, y: coordenadas de los puntos
    - labels: etiquetas de título para cada punto
    - ax: eje de matplotlib
    - font_size: tamaño de fuente para las etiquetas
    - sample_step: paso para mostrar etiquetas (1 = todas, >1 = muestreo)
    """
    # Mostrar todas las etiquetas o un muestreo
    for i in range(0, len(x), sample_step):
        ax.annotate(labels.iloc[i], 
                   (x[i], y[i]), 
                   fontsize=font_size, 
                   alpha=0.6,
                   xytext=(5, 5), 
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

# =====================================================================
# GRÁFICO 1: PCA
# =====================================================================
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

# =====================================================================
# GRÁFICO 2: t-SNE
# =====================================================================
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

# =====================================================================
# GRÁFICO 3: K-Means Clustering
# =====================================================================
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

# =====================================================================
# GRÁFICO 4: Agglomerative Clustering
# =====================================================================
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

# =====================================================================
# GRÁFICO 5: DBSCAN
# =====================================================================
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

# =====================================================================
# GRÁFICO 6: OPTICS Clustering
# =====================================================================
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

print("\n✅ Todos los gráficos han sido generados exitosamente.")
print("💡 Nota: El parámetro 'sample_step=10' en add_titulo_labels muestra 1 de cada 10 etiquetas para evitar saturación visual.")
print("   Puedes ajustar 'sample_step' a 1 para mostrar TODAS las etiquetas, o a un número mayor para menos etiquetas.")