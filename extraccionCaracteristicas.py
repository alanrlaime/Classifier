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

def procesar_video_columnas(url, n_frames_salto):
    ydl_opts = {'format': 'worstvideo[ext=mp4]/mp4', 'quiet': True}
    
    metricas_acumuladas = []
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_url = info['url']
            
        cap = cv2.VideoCapture(video_url)
        frame_count = 0
        frame_anterior = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            if frame_count % n_frames_salto == 0:
                res = extraer_metricas_frame(frame, frame_anterior)
                metricas_acumuladas.append(res)
                frame_anterior = frame.copy() # Guardar para la métrica de movimiento
                
            frame_count += 1
            
        cap.release()
        
        if metricas_acumuladas:
            # Convertimos la lista de diccionarios a un DataFrame temporal para promediar fácilmente
            df_temp = pd.DataFrame(metricas_acumuladas)
            valores_promedio = df_temp.mean().to_dict()
            return valores_promedio
        return None
        
    except Exception as e:
        print(f"Error en URL {url}: {e}")
        return None

# --- EJECUCIÓN ---
def main():
    N_FRAMES = 120  # Aprox. 1 frame cada 4 segundos
    df = pd.read_csv("dataset_youtube.csv")
    
    filas_finales = []
    
    for index, row in df.iterrows():
        print(f"Analizando [{index+1}/{len(df)}]: {row['Titulo']}")
        dict_metricas = procesar_video_columnas(row['URL'], N_FRAMES)
        
        # Base de la fila
        fila = {
            'id': row['id'],
            'Titulo': row['Titulo'],
            'URL': row['URL']
        }
        
        # Si la extracción fue exitosa, unimos las métricas como columnas individuales
        if dict_metricas:
            fila.update(dict_metricas)
        else:
            # Columnas vacías en caso de error
            fila.update({k: np.nan for k in ['brillo', 'contraste', 'densidad_bordes', 'movimiento']})
            
        filas_finales.append(fila)
        
    # Guardar el CSV final completamente distribuido
    df_salida = pd.DataFrame(filas_finales)
    df_salida.to_csv("extraccion_dataset.csv", index=False)
    print("\n¡Proceso finalizado! Archivo guardado como 'extraccion_dataset.csv'")

main()