import cv2
import yt_dlp
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.cluster import OPTICS
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import mediapipe as mp  
# --- CONFIGURACIÓN LBP ---
# Parámetros para el radio y número de puntos vecinos
METHOD = 'uniform'
RADIUS = 3
N_POINTS = 8 * RADIUS

# --- CONFIGURACIÓN GENERAL ---
youtube_url = 'https://www.youtube.com/watch?v=r5BvfctOXDA'
model_path = 'face_detector.tflite'
output_folder = 'clusters_lbp'
segundos_a_procesar = 100
skip_frames = 5

if not os.path.exists(output_folder): os.makedirs(output_folder)

def get_lbp_descriptor(image):
    """Convierte un recorte de rostro en un histograma LBP (Embedding Primitivo)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (100, 100)) # Normalizar tamaño para que los histogramas sean comparables
    
    # Calcular LBP
    lbp = local_binary_pattern(gray, N_POINTS, RADIUS, METHOD)
    
    # Crear histograma (este será nuestro 'embedding')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), range=(0, N_POINTS + 2))
    
    # Normalizar histograma
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# 1. Obtener Stream
ydl_opts = {'format': 'best[height<=360]', 'quiet': True}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    stream_url = ydl.extract_info(youtube_url, download=False)['url']

# 2. Configurar Detector (MediaPipe para localizar el rostro)
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)

embeddings_list = []
faces_list = []

with vision.FaceDetector.create_from_options(options) as detector:
    cap = cv2.VideoCapture(stream_url)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0

    print("📸 Fase 1: Extrayendo descriptores LBP...")

    while cap.isOpened() and frame_idx < (fps * segundos_a_procesar):
        success, frame = cap.read()
        if not success: break

        if frame_idx % skip_frames == 0:
            timestamp_ms = int((frame_idx / fps) * 1000)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            result = detector.detect_for_video(mp_image, timestamp_ms)

            if result.detections:
                for detection in result.detections:
                    bbox = detection.bounding_box
                    # Extraer recorte
                    x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
                    # Asegurar que el recorte esté dentro de la imagen
                    x, y = max(0, x), max(0, y)
                    face_crop = frame[y:y+h, x:x+w]
                    
                    if face_crop.size > 0:
                        # Generar Descriptor LBP
                        lbp_desc = get_lbp_descriptor(face_crop)
                        embeddings_list.append(lbp_desc)
                        faces_list.append(face_crop)

        frame_idx += 1

cap.release()

# --- FASE 2: CLUSTERING OPTICS ---
if len(embeddings_list) > 0:
    print(f"🧬 Fase 2: Aplicando OPTICS sobre {len(embeddings_list)} rostros...")
    X = np.array(embeddings_list)
    
    # Ajustamos min_samples. OPTICS agrupará histogramas similares.
    clust = OPTICS(min_samples=3, metric='cosine').fit(X) # 'cosine' suele funcionar mejor para histogramas
    labels = clust.labels_

    for i, label in enumerate(labels):
        cluster_name = f"Persona_{label}" if label != -1 else "Ruido"
        path = os.path.join(output_folder, cluster_name)
        if not os.path.exists(path): os.makedirs(path)
        cv2.imwrite(os.path.join(path, f"face_{i}.jpg"), faces_list[i])

    print(f"\n✅ Proceso completado. Grupos creados en '{output_folder}'.")
else:
    print("No se encontraron rostros.")