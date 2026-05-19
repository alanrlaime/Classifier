# algoritmo.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle

class AlgoritmoClustering:
    """
    Clase que encapsula el algoritmo de K-Means clustering
    """
    
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2, random_state=random_state)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        self.is_fitted = False
        
    def cargar_datos(self, archivo='extraccion_dataset.csv'):
        """
        Carga y preprocesa los datos
        """
        try:
            df = pd.read_csv(archivo)
            print(f"✅ Datos cargados desde {archivo}")
        except FileNotFoundError:
            print(f"⚠️ Archivo {archivo} no encontrado. Creando datos sintéticos...")
            from sklearn.datasets import make_blobs
            X_synthetic, _ = make_blobs(n_samples=300, n_features=6, centers=3, random_state=42)
            df = pd.DataFrame(X_synthetic, columns=[f'num_col_{i}' for i in range(6)])
            df.insert(0, 'Col_Inutil_1', 'A')
            df.insert(1, 'Col_Inutil_2', 'B')
            df.insert(2, 'Col_Inutil_3', 'C')
            df.insert(3, 'Titulo', np.random.choice(['Categoría A', 'Categoría B', 'Categoría C'], size=300))
        
        # Guardar columna Titulo
        self.labels_titulo = df['Titulo']
        
        # Seleccionar columnas numéricas
        X_numeric = df.iloc[:, 13:].select_dtypes(include=[np.number])
        idx_validos = X_numeric.dropna().index
        self.X_numeric = X_numeric.loc[idx_validos]
        self.labels_titulo = self.labels_titulo.loc[idx_validos]
        self.datos_originales = df.loc[idx_validos]
        
        print(f"📊 Columnas numéricas: {list(self.X_numeric.columns)}")
        print(f"📈 Total de muestras: {len(self.X_numeric)}")
        
        return self
    
    def preprocesar(self):
        """
        Escala los datos
        """
        self.X_scaled = self.scaler.fit_transform(self.X_numeric)
        print("✅ Datos escalados correctamente")
        return self
    
    def reducir_dimensionalidad(self):
        """
        Aplica PCA para reducción a 2D
        """
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        print(f"✅ PCA completado: {self.pca.explained_variance_ratio_.sum()*100:.2f}% de varianza explicada")
        return self
    
    def aplicar_clustering(self):
        """
        Aplica K-Means clustering
        """
        self.labels_kmeans = self.kmeans.fit_predict(self.X_pca)
        self.is_fitted = True
        print(f"✅ Clustering K-Means completado con {self.n_clusters} clusters")
        
        # Mostrar distribución de clusters
        unique, counts = np.unique(self.labels_kmeans, return_counts=True)
        for cluster, count in zip(unique, counts):
            print(f"   Cluster {cluster}: {count} muestras ({count/len(self.labels_kmeans)*100:.1f}%)")
        
        return self
    
    def get_centroides(self):
        """
        Retorna los centroides en el espacio PCA
        """
        if not self.is_fitted:
            raise ValueError("El modelo no ha sido entrenado aún")
        return self.kmeans.cluster_centers_
    
    def get_info_muestra(self, idx):
        """
        Retorna toda la información de una muestra específica
        """
        if idx >= len(self.datos_originales):
            raise IndexError("Índice fuera de rango")
        
        muestra = self.datos_originales.iloc[idx]
        info = {
            'Índice': idx,
            'Título': self.labels_titulo.iloc[idx],
            'Cluster': int(self.labels_kmeans[idx]),
            'Coordenadas PCA': {
                'x': float(self.X_pca[idx, 0]),
                'y': float(self.X_pca[idx, 1])
            },
            'Valores numéricos': muestra[self.X_numeric.columns].to_dict(),
            'Datos completos': muestra.to_dict()
        }
        return info
    
    def guardar_modelo(self, archivo='modelo_clustering.pkl'):
        """
        Guarda el modelo entrenado en un archivo
        """
        with open(archivo, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ Modelo guardado en {archivo}")
    
    @staticmethod
    def cargar_modelo(archivo='modelo_clustering.pkl'):
        """
        Carga un modelo guardado
        """
        with open(archivo, 'rb') as f:
            modelo = pickle.load(f)
        print(f"✅ Modelo cargado desde {archivo}")
        return modelo

# Ejecución directa para pruebas
if __name__ == "__main__":
    print("🚀 Ejecutando algoritmo de clustering...")
    modelo = AlgoritmoClustering(n_clusters=3)
    modelo.cargar_datos().preprocesar().reducir_dimensionalidad().aplicar_clustering()
    modelo.guardar_modelo()
    print("\n🎯 Ejemplo de información de una muestra:")
    print(modelo.get_info_muestra(0))