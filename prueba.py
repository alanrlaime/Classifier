import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time

# --- CONFIGURACIÓN ---
input_video_path = 'video.mp4'  # Cambia por el nombre de tu archivo
output_video_path = 'video2.mp4'
model_path = 'face_detector.tflite'
segundos_a_grabar = 100

# Verificar que el modelo existe
if not os.path.exists(model_path):
    print(f"Error crítico: No se encontró {model_path} en la carpeta.")
    exit()

# Configurar MediaPipe Tasks
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO)

with vision.FaceDetector.create_from_options(options) as detector:
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir el video de entrada.")
        exit()

    # Metadatos para el guardado
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = fps * segundos_a_grabar

    # Configurar el escritor (Codec MP4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Procesando {segundos_a_grabar} segundos de video... Por favor espera.")

    for n in range(total_frames):
        success, frame = cap.read()
        if not success:
            break

        # Timestamp requerido por MediaPipe Tasks
        timestamp_ms = int((n / fps) * 1000)

        # Preparar imagen para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detección
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)

        # Dibujar rectángulos en el frame (se guardarán en el archivo)
        if detection_result.detections:
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
                
                # Dibujar directamente sobre el frame original
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                
                score = int(detection.categories[0].score * 100)
                cv2.putText(frame, f"Rostro: {score}%", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Escribir frame procesado al disco
        out.write(frame)

        # Opcional: Imprimir progreso en consola
        if n % fps == 0:
            print(f"Progreso: {n // fps}s procesados...")

    # Liberar recursos
    cap.release()
    out.release()
    print(f"✅ Proceso terminado. El video se guardó en: {output_video_path}")