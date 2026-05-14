import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# --- CONFIGURACIÓN ---
# Asegúrate de que este archivo esté en tu carpeta
model_path = 'face_detector.tflite' 

# Configurar las opciones del detector
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO) # Optimizado para webcam

with vision.FaceDetector.create_from_options(options) as detector:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convertir BGR (OpenCV) a RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Necesitamos el timestamp en milisegundos para el modo VIDEO
        timestamp = int(time.time() * 1000)
        
        # Detectar rostros
        detection_result = detector.detect_for_video(mp_image, timestamp)

        # Dibujar resultados
        if detection_result.detections:
            for detection in detection_result.detections:
                # Obtener la caja delimitadora (Bounding Box)
                bbox = detection.bounding_box
                start_point = int(bbox.origin_x), int(bbox.origin_y)
                end_point = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
                
                # Dibujar rectángulo verde
                cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
                
                # Mostrar el score (confianza)
                score = round(detection.categories[0].score, 2)
                cv2.putText(frame, f"Conf: {score}", (start_point[0], start_point[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('MediaPipe Tasks - Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == 27: # Presiona ESC para salir
            break

    cap.release()
    cv2.destroyAllWindows()