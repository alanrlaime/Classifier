# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import pickle

# Importar nuestro algoritmo
from algoritmo import AlgoritmoClustering

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Clustering Interactivo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.title("Análisis de Clustering Interactivo")
st.markdown("### Visualización de clusters")

# Sidebar para controles
with st.sidebar:
    st.header("Configuración")
    
    # Cargar o entrenar modelo
    st.subheader("1. Modelo")
    modelo_opcion = st.radio(
        "Selecciona una opción:",
        ["Cargar modelo existente", "Entrenar nuevo modelo"],
        index=1
    )
    
    if modelo_opcion == "Entrenar nuevo modelo":
        n_clusters = st.slider("Número de clusters (K)", 2, 8, 3, 1)
        archivo_datos = st.text_input("Archivo de datos", "extraccion_dataset.csv")
        
        if st.button("Entrenar modelo", use_container_width=True):
            with st.spinner("Entrenando modelo..."):
                modelo = AlgoritmoClustering(n_clusters=n_clusters)
                modelo.cargar_datos(archivo_datos).preprocesar().reducir_dimensionalidad().aplicar_clustering()
                modelo.guardar_modelo()
                st.session_state['modelo'] = modelo
                st.success("Modelo entrenado y guardado.")
    else:
        archivo_modelo = st.text_input("Archivo del modelo", "modelo_clustering.pkl")
        if st.button("Cargar modelo", use_container_width=True):
            try:
                modelo = AlgoritmoClustering.cargar_modelo(archivo_modelo)
                st.session_state['modelo'] = modelo
                st.success("Modelo cargado exitosamente.")
            except FileNotFoundError:
                st.error(f"No se encontró el archivo {archivo_modelo}")
    
    # Configuración de visualización
    st.subheader("2. Visualización")
    punto_size = st.slider("Tamaño de puntos", 5, 20, 10)
    opacidad = st.slider("Opacidad", 0.3, 1.0, 0.7)
    mostrar_centroides = st.checkbox("Mostrar centroides", value=True)
    mostrar_etiquetas = st.checkbox("Mostrar etiquetas de títulos", value=False)
    
    st.markdown("---")
    st.markdown("**Instrucciones:**")
    st.markdown("- Usa el selector para ver información de cada punto")
    st.markdown("- Pasa el mouse sobre los puntos para ver datos básicos")
    st.markdown("- Ajusta los controles para personalizar la vista")

# Inicializar modelo en session state
if 'modelo' not in st.session_state:
    st.session_state['modelo'] = None
if 'punto_seleccionado' not in st.session_state:
    st.session_state['punto_seleccionado'] = None

# Verificar que el modelo esté cargado
if st.session_state['modelo'] is None:
    st.warning("Por favor, carga o entrena un modelo en la barra lateral izquierda")
    st.stop()

modelo = st.session_state['modelo']

# Layout principal con dos columnas
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Visualización del Clustering")
    
    # Crear gráfico interactivo con Plotly
    df_plot = pd.DataFrame({
        'PCA1': modelo.X_pca[:, 0],
        'PCA2': modelo.X_pca[:, 1],
        'Cluster': modelo.labels_kmeans.astype(str),
        'Título': modelo.labels_titulo,
        'Índice': range(len(modelo.X_pca))
    })
    
    # Crear figura
    fig = go.Figure()
    
    # Añadir puntos por cluster
    for cluster in sorted(df_plot['Cluster'].unique()):
        df_cluster = df_plot[df_plot['Cluster'] == cluster]
        fig.add_trace(go.Scatter(
            x=df_cluster['PCA1'],
            y=df_cluster['PCA2'],
            mode='markers+text' if mostrar_etiquetas else 'markers',
            name=f'Cluster {cluster}',
            text=df_cluster['Título'] if mostrar_etiquetas else None,
            textposition="top center",
            marker=dict(
                size=punto_size,
                opacity=opacidad,
                symbol='circle',
                line=dict(width=1, color='black')
            ),
            hovertemplate=(
                f'<b>Índice:</b> %{{customdata[0]}}<br>' +
                f'<b>Título:</b> %{{customdata[1]}}<br>' +
                f'<b>Cluster:</b> {cluster}<br>' +
                f'<b>PCA1:</b> %{{x:.3f}}<br>' +
                f'<b>PCA2:</b> %{{y:.3f}}<br>' +
                '<extra></extra>'
            ),
            customdata=np.stack((df_cluster['Índice'], df_cluster['Título']), axis=-1)
        ))
    
    # Añadir centroides
    if mostrar_centroides:
        centroides = modelo.get_centroides()
        fig.add_trace(go.Scatter(
            x=centroides[:, 0],
            y=centroides[:, 1],
            mode='markers',
            name='Centroides',
            marker=dict(
                size=15,
                symbol='x',
                color='red',
                line=dict(width=3, color='darkred')
            ),
            hovertemplate='<b>Centroide</b><br>PCA1: %{x:.3f}<br>PCA2: %{y:.3f}<extra></extra>'
        ))
    
    # Configurar layout
    fig.update_layout(
        title={
            'text': "Proyección PCA de los datos coloreados por cluster",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Arial'}
        },
        xaxis_title="Componente Principal 1",
        yaxis_title="Componente Principal 2",
        hovermode='closest',
        plot_bgcolor='rgba(240,240,240,0.8)',
        paper_bgcolor='white',
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        )
    )
    
    # Mostrar gráfico
    st.plotly_chart(fig, use_container_width=True, key="clustering_plot")
    
    # Selector de punto
    st.subheader("Selección de muestra")
    indices_disponibles = list(range(len(modelo.X_pca)))
    
    # Crear opciones con formato amigable
    opciones = []
    for i in indices_disponibles:
        opciones.append(f"Muestra {i} - {modelo.labels_titulo.iloc[i][:50]}")
    
    idx_seleccionado = st.selectbox(
        "Selecciona una muestra para ver su información detallada:",
        indices_disponibles,
        format_func=lambda x: f"Muestra {x} - {modelo.labels_titulo.iloc[x][:50]}"
    )
    
    if idx_seleccionado is not None:
        st.session_state['punto_seleccionado'] = idx_seleccionado

with col2:
    st.header("Información de la muestra")
    
    if st.session_state['punto_seleccionado'] is not None:
        idx = st.session_state['punto_seleccionado']
        info = modelo.get_info_muestra(idx)
        datos_muestra = modelo.datos_originales.iloc[idx]
        
        col2a, col2b, col2c = st.columns(3)
        with col2a:
            st.metric("Índice", info['Índice'])
        with col2b:
            st.metric("Cluster", info['Cluster'])
        with col2c:
            st.metric("Total muestras", len(modelo.X_pca))
        
        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Contenido", "Características", "Posición", "Todos los datos"])
        
        with tab1:
            st.subheader("Información General")
            
            col_cont1, col_cont2 = st.columns([2, 1])
            
            with col_cont1:
                st.markdown(f"""
                <div class="info-box">
                    <b style="font-size: 16px;">Título del Video</b><br>
                    <span style="font-size: 14px; font-weight: bold;">{info['Título']}</span>
                </div>
                """, unsafe_allow_html=True)
                
                if 'Canal' in datos_muestra:
                    st.markdown(f"**Canal:** {datos_muestra['Canal']}")
                
                if 'Fecha_Subida' in datos_muestra:
                    st.markdown(f"**Fecha de subida:** {datos_muestra['Fecha_Subida']}")
                
                if 'Duracion_Segundos' in datos_muestra:
                    duracion_min = int(datos_muestra['Duracion_Segundos'] / 60)
                    duracion_seg = int(datos_muestra['Duracion_Segundos'] % 60)
                    st.markdown(f"**Duración:** {duracion_min}m {duracion_seg}s")
                
                if 'Resolucion' in datos_muestra:
                    st.markdown(f"**Resolución:** {datos_muestra['Resolucion']}")
            
            with col_cont2:
                if 'Thumbnail_URL' in datos_muestra and pd.notna(datos_muestra['Thumbnail_URL']):
                    try:
                        st.image(datos_muestra['Thumbnail_URL'], use_container_width=True, caption="Portada")
                    except:
                        st.warning("No se pudo cargar la portada")
                elif 'Ruta_Portada' in datos_muestra and pd.notna(datos_muestra['Ruta_Portada']):
                    try:
                        st.image(datos_muestra['Ruta_Portada'], use_container_width=True, caption="Portada local")
                    except:
                        st.info("Imagen local no disponible")
                else:
                    st.info("Sin portada disponible")
            
            st.markdown("---")
            st.subheader("Enlaces")
            
            if 'URL' in datos_muestra and pd.notna(datos_muestra['URL']):
                st.markdown(f"[🔗 Ver video en YouTube]({datos_muestra['URL']})", unsafe_allow_html=True)
                st.code(datos_muestra['URL'], language="text")
            
            if 'Thumbnail_URL' in datos_muestra and pd.notna(datos_muestra['Thumbnail_URL']):
                st.markdown(f"[🖼️ Ver portada completa]({datos_muestra['Thumbnail_URL']})", unsafe_allow_html=True)
            
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                if 'Video_ID' in datos_muestra:
                    st.markdown(f"**ID del Video:** `{datos_muestra['Video_ID']}`")
            
            with col_info2:
                if 'FPS' in datos_muestra:
                    st.markdown(f"**FPS:** {datos_muestra['FPS']}")
        
        with tab2:
            st.subheader("Características Visuales")
            
            caracteristicas_visuales = {
                'brillo': 'Brillo',
                'contraste': 'Contraste',
                'densidad_bordes': 'Densidad de bordes',
                'movimiento': 'Movimiento'
            }
            
            col_car1, col_car2 = st.columns(2)
            
            for i, (col_name, label) in enumerate(caracteristicas_visuales.items()):
                if col_name in datos_muestra:
                    valor = float(datos_muestra[col_name])
                    if i % 2 == 0:
                        with col_car1:
                            st.metric(label, f"{valor:.2f}")
                    else:
                        with col_car2:
                            st.metric(label, f"{valor:.2f}")
            
            st.markdown("---")
            st.subheader("Información Técnica")
            
            tecnica_info = {}
            if 'Formato' in datos_muestra:
                tecnica_info['Formato'] = datos_muestra['Formato']
            if 'FileSize_Bytes' in datos_muestra:
                size_mb = datos_muestra['FileSize_Bytes'] / (1024 * 1024)
                tecnica_info['Tamaño'] = f"{size_mb:.2f} MB"
            
            for key, value in tecnica_info.items():
                st.markdown(f"**{key}:** {value}")
            
            st.markdown("---")
            st.subheader("Características de Color")
            
            colores_disponibles = [col for col in datos_muestra.index if col.startswith('color_')]
            
            if colores_disponibles:
                col_color1, col_color2 = st.columns(2)
                
                for i, col_name in enumerate(colores_disponibles):
                    valor = float(datos_muestra[col_name])
                    if i % 2 == 0:
                        with col_color1:
                            st.metric(col_name.upper(), f"{valor:.4f}")
                    else:
                        with col_color2:
                            st.metric(col_name.upper(), f"{valor:.4f}")
        
        with tab3:
            st.subheader("Análisis de Posición en Clustering")
            
            col_pos1, col_pos2 = st.columns(2)
            
            with col_pos1:
                st.markdown("**Coordenadas PCA:**")
                st.markdown(f"- **Componente 1:** `{info['Coordenadas PCA']['x']:.6f}`")
                st.markdown(f"- **Componente 2:** `{info['Coordenadas PCA']['y']:.6f}`")
            
            with col_pos2:
                cluster_asignado = info['Cluster']
                st.markdown(f"**Cluster asignado:** `{cluster_asignado}`")
                
                distancia_centroide = np.sqrt(
                    (info['Coordenadas PCA']['x'] - modelo.kmeans.cluster_centers_[cluster_asignado, 0])**2 + 
                    (info['Coordenadas PCA']['y'] - modelo.kmeans.cluster_centers_[cluster_asignado, 1])**2
                )
                st.markdown(f"**Distancia al centroide:** `{distancia_centroide:.4f}`")
            
            st.markdown("---")
            st.subheader("Distancias a todos los centroides")
            
            distancias = []
            for i in range(len(modelo.kmeans.cluster_centers_)):
                dist = np.sqrt(
                    (info['Coordenadas PCA']['x'] - modelo.kmeans.cluster_centers_[i, 0])**2 + 
                    (info['Coordenadas PCA']['y'] - modelo.kmeans.cluster_centers_[i, 1])**2
                )
                distancias.append(dist)
            
            df_distancias = pd.DataFrame({
                'Cluster': [f'Cluster {i}' for i in range(len(distancias))],
                'Distancia': distancias
            })
            
            fig_dist = px.bar(df_distancias, x='Cluster', y='Distancia', 
                             title="Distancia a cada centroide",
                             color='Distancia',
                             color_continuous_scale='Viridis')
            st.plotly_chart(fig_dist, use_container_width=True)
            
            st.success(f"Esta muestra pertenece al **Cluster {cluster_asignado}**")
        
        with tab4:
            st.subheader("Todos los datos de la muestra")
            
            df_todos = pd.DataFrame({
                'Campo': info['Datos completos'].keys(),
                'Valor': info['Datos completos'].values()
            })
            
            st.dataframe(df_todos, use_container_width=True, height=500)
            
            col_export1, col_export2 = st.columns(2)
            with col_export1:
                csv = df_todos.to_csv(index=False)
                st.download_button(
                    label="Descargar como CSV",
                    data=csv,
                    file_name=f"muestra_{idx}.csv",
                    mime="text/csv"
                )
            with col_export2:
                json_str = df_todos.set_index('Campo')['Valor'].to_json(indent=2)
                st.download_button(
                    label="Descargar como JSON",
                    data=json_str,
                    file_name=f"muestra_{idx}.json",
                    mime="application/json"
                )
        
        st.markdown("---")
        if st.button("Limpiar selección", use_container_width=True):
            st.session_state['punto_seleccionado'] = None
            st.rerun()
        
    else:
        st.info("Selecciona una muestra del menú desplegable para ver su información detallada")
        st.markdown("""
        ### Instrucciones:
        1. Usa el selector de la izquierda para elegir una muestra
        2. La información completa aparecerá aquí con múltiples pestañas
        3. Tab "Contenido": Información general, enlaces y portada
        4. Tab "Características": Datos visuales y técnicos
        5. Tab "Posición": Análisis de ubicación en clustering
        6. Tab "Todos los datos": Vista completa y descargable
        """)

# Sección de estadísticas adicionales
st.markdown("---")
st.header("Estadísticas del Clustering")

col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)

with col_stats1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Número de clusters", modelo.n_clusters)
    st.markdown('</div>', unsafe_allow_html=True)

with col_stats2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total de muestras", len(modelo.X_pca))
    st.markdown('</div>', unsafe_allow_html=True)

with col_stats3:
    varianza_explicada = modelo.pca.explained_variance_ratio_.sum() * 100
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Varianza explicada por PCA", f"{varianza_explicada:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

with col_stats4:
    # Calcular silueta promedio aproximada
    try:
        from sklearn.metrics import silhouette_score
        sil_score = silhouette_score(modelo.X_pca, modelo.labels_kmeans)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Coeficiente de Silueta", f"{sil_score:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    except:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Coeficiente de Silueta", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)

# Distribución de clusters
st.subheader("Distribución de muestras por cluster")
df_clusters = pd.DataFrame({
    'Cluster': [f'Cluster {i}' for i in range(modelo.n_clusters)],
    'Cantidad': [np.sum(modelo.labels_kmeans == i) for i in range(modelo.n_clusters)]
})

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    fig_bar = px.bar(df_clusters, x='Cluster', y='Cantidad', 
                     title="Número de muestras por cluster",
                     color='Cluster',
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

with col_chart2:
    fig_pie = px.pie(df_clusters, values='Cantidad', names='Cluster',
                     title="Proporción por cluster",
                     color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig_pie, use_container_width=True)

# Matriz de correlación (opcional)
with st.expander("Ver matriz de correlación de características numéricas"):
    try:
        fig_corr = px.imshow(modelo.X_numeric.corr(), 
                             text_auto=True, 
                             aspect="auto",
                             title="Matriz de Correlación",
                             color_continuous_scale='RdBu')
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
    except Exception as e:
        st.warning(f"No se pudo generar la matriz de correlación: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 20px;">
    <p>Aplicación interactiva de clustering | Desarrollada con Streamlit, Plotly y scikit-learn</p>
    <p>Los puntos coloreados representan los clusters identificados por K-Means</p>
    <p>Selecciona una muestra del menú desplegable para ver todos sus detalles</p>
</div>
""", unsafe_allow_html=True)