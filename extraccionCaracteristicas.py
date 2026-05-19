import cv2
import pandas as pd
import numpy as np
import yt_dlp

def extraer_metricas_frame(frame, frame_anterior=None):
    """Extrae características visuales puramente matemáticas sin Deep Learning"""
    # 1. Redimensionar y convertir a escala de grises para métricas estructurales
    frame_pequeno = cv2.resize(frame, (320, 240))
    gris = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2HSV)
    
    # --- MÉTRICAS ESTADÍSTICAS ---
    brillo = np.mean(gris)
    contraste = np.std(gris)
    
    # --- DENSIDAD DE BORDES (Estructura/Texto) ---
    # Sobel detecta cambios drásticos de intensidad (bordes de letras o figuras)
    sobel_x = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)
    magnitud_bordes = np.sqrt(sobel_x**2 + sobel_y**2)
    densidad_bordes = np.mean(magnitud_bordes)
    
    # --- VARIACIÓN TEMPORAL (Movimiento) ---
    movimiento = 0.0
    if frame_anterior is not None:
        gris_anterior = cv2.cvtColor(cv2.resize(frame_anterior, (320, 240)), cv2.COLOR_BGR2GRAY)
        # Diferencia absoluta entre el frame actual y el anterior
        diferencia = cv2.absdiff(gris, gris_anterior)
        movimiento = np.mean(diferencia)
        
    # --- HISTOGRAMA SIMPLIFICADO (Color) ---
    # Solo 4 bins para Tono (Hue) y 4 para Saturación para no saturar de columnas
    hist_h = cv2.calcHist([hsv], [0], None, [4], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [4], [0, 256]).flatten()
    
    # Normalizar histogramas
    hist_h = hist_h / (hist_h.sum() + 1e-5)
    hist_s = hist_s / (hist_s.sum() + 1e-5)
    
    # Empaquetamos todo en un diccionario plano
    caracteristicas = {
        'brillo': round(float(brillo), 2),
        'contraste': round(float(contraste), 2),
        'densidad_bordes': round(float(densidad_bordes), 2),
        'movimiento': round(float(movimiento), 2),
        'color_h1': round(float(hist_h[0]), 4),
        'color_h2': round(float(hist_h[1]), 4),
        'color_h3': round(float(hist_h[2]), 4),
        'color_h4': round(float(hist_h[3]), 4),
        'color_s1': round(float(hist_s[0]), 4),
        'color_s2': round(float(hist_s[1]), 4),
        'color_s3': round(float(hist_s[2]), 4),
        'color_s4': round(float(hist_s[3]), 4),
    }
    return caracteristicas

def obtener_duracion_video(url):
    """Obtiene la duración del video en segundos sin descargarlo"""
    ydl_opts = {'format': 'worstvideo[ext=mp4]/mp4', 'quiet': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            duracion = info.get('duration', 0)
            return duracion
    except Exception as e:
        print(f"Error obteniendo duración: {e}")
        return None

def procesar_video_columnas(url, n_frames_salto, max_duracion_segundos=1200):
    """
    Procesa el video y extrae métricas
    max_duracion_segundos: tiempo máximo a procesar en segundos (default 1200 = 20 minutos)
    """
    ydl_opts = {'format': 'worstvideo[ext=mp4]/mp4', 'quiet': True}
    
    metricas_acumuladas = []
    
    try:
        # Obtener información del video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_url = info['url']
            duracion_total = info.get('duration', 0)
            
            # Validar si el video existe y tiene duración
            if duracion_total is None or duracion_total == 0:
                print(f"  ⚠️ No se pudo determinar la duración del video")
                return None
        
        # Verificar si el video es largo (>20 minutos)
        es_video_largo = duracion_total > max_duracion_segundos
        
        if es_video_largo:
            minutos = duracion_total / 60
            print(f"  📹 Video largo detectado ({minutos:.1f} minutos). Procesando solo primeros {max_duracion_segundos/60} minutos...")
        else:
            print(f"  📹 Duración del video: {duracion_total/60:.1f} minutos")
        
        cap = cv2.VideoCapture(video_url)
        
        # Obtener FPS del video
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # Valor por defecto si no se puede obtener
        
        # Calcular cuántos frames corresponden a los primeros X segundos
        if es_video_largo:
            max_frames_a_procesar = int(fps * max_duracion_segundos)
        else:
            max_frames_a_procesar = float('inf')  # Procesar todo el video
        
        frame_count = 0
        frame_anterior = None
        frames_procesados = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break
            
            # Si hemos superado el límite de frames (para videos largos), salir
            if frame_count >= max_frames_a_procesar:
                print(f"  ✅ Límite de {max_duracion_segundos} segundos alcanzado")
                break
            
            if frame_count % n_frames_salto == 0:
                res = extraer_metricas_frame(frame, frame_anterior)
                metricas_acumuladas.append(res)
                frame_anterior = frame.copy()
                frames_procesados += 1
                
            frame_count += 1
            
        cap.release()
        
        # Validar si se extrajeron suficientes métricas
        if len(metricas_acumuladas) == 0:
            print(f"  ❌ No se extrajeron métricas del video")
            return None
        
        # Validar datos vacíos o nulos en las métricas
        metricas_validas = []
        for m in metricas_acumuladas:
            # Verificar que todas las métricas tengan valores válidos
            if all(v is not None and not pd.isna(v) for v in m.values()):
                metricas_validas.append(m)
        
        if len(metricas_validas) == 0:
            print(f"  ❌ Todas las métricas extraídas contenían valores nulos")
            return None
        
        # Convertir la lista de diccionarios a un DataFrame temporal para promediar fácilmente
        df_temp = pd.DataFrame(metricas_validas)
        valores_promedio = df_temp.mean().to_dict()
        
        print(f"  ✅ Procesado exitosamente: {frames_procesados} frames analizados")
        return valores_promedio
        
    except Exception as e:
        print(f"  ❌ Error en procesamiento: {e}")
        return None

def validar_fila_csv(row):
    """Valida que una fila del CSV tenga datos completos y no esté vacía"""
    # Verificar campos obligatorios
    campos_requeridos = ['ID', 'Titulo', 'URL']
    
    for campo in campos_requeridos:
        if campo not in row.index:
            print(f"  ⚠️ Campo requerido '{campo}' no encontrado")
            return False
        
        valor = row[campo]
        # Verificar que no sea nulo, vacío o solo espacios
        if pd.isna(valor) or str(valor).strip() == '':
            print(f"  ⚠️ Campo '{campo}' vacío o nulo")
            return False
    
    # Verificar que la URL sea válida
    url = str(row['URL']).strip()
    if not (url.startswith('http://') or url.startswith('https://')):
        print(f"  ⚠️ URL inválida: {url[:50]}...")
        return False
    
    return True


N_FRAMES = 120  # Aprox. 1 frame cada 4 segundos (asumiendo 30 fps)
MAX_DURACION_SEGUNDOS = 600  # 10 minutos en segundos

# Leer el CSV original
df = pd.read_csv("dataset_youtube.csv")

print(f"📊 Total de videos a analizar: {len(df)}")
print("="*50)

filas_finales = []
videos_procesados = 0
videos_con_error = 0
videos_vacios = 0

for index, row in df.iterrows():
    print(f"\n🔍 Analizando [{index+1}/{len(df)}]: {row.get('Titulo', 'Sin título')[:50]}...")
    
    # Validar que la fila tenga datos completos
    if not validar_fila_csv(row):
        print(f"  ⚠️ Video {index+1} omitido - Datos incompletos")
        videos_vacios += 1
        continue
    
    # Extraer métricas del video
    dict_metricas = procesar_video_columnas(row['URL'], N_FRAMES, MAX_DURACION_SEGUNDOS)
    
    # Crear fila con todas las columnas originales
    fila = row.to_dict()  # Mantener todas las columnas originales del CSV
    
    # Si la extracción fue exitosa, agregamos las métricas como columnas individuales
    if dict_metricas:
        fila.update(dict_metricas)
        videos_procesados += 1
    else:
        # Agregar columnas vacías para las métricas en caso de error
        columnas_metricas = ['brillo', 'contraste', 'densidad_bordes', 'movimiento',
                            'color_h1', 'color_h2', 'color_h3', 'color_h4',
                            'color_s1', 'color_s2', 'color_s3', 'color_s4']
        for col in columnas_metricas:
            fila[col] = np.nan
        videos_con_error += 1
    
    filas_finales.append(fila)

# Crear DataFrame final
df_salida = pd.DataFrame(filas_finales)

# Guardar el CSV final
df_salida.to_csv("extraccion_dataset.csv", index=False)

# Mostrar estadísticas finales
print("\n" + "="*50)
print("📊 ESTADÍSTICAS FINALES:")
print(f"  ✅ Videos procesados exitosamente: {videos_procesados}")
print(f"  ❌ Videos con error: {videos_con_error}")
print(f"  ⚠️ Videos omitidos (datos incompletos): {videos_vacios}")
print(f"  📁 Total de videos en dataset original: {len(df)}")
print(f"  📄 Archivo guardado como 'extraccion_dataset.csv'")
print("="*50)
