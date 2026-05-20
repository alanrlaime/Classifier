import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import yt_dlp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from skimage.feature import local_binary_pattern
from sklearn.cluster import OPTICS

METODO = 'uniform'
RADIO = 3
N_PUNTOS = 8 * RADIO
OPCIONES_YDL_POR_DEFECTO = {'format': 'best[height<=360]', 'quiet': True}
MIN_MUESTRAS_POR_DEFECTO = 3
METRICA_POR_DEFECTO = 'cosine'


def obtener_descriptor_lbp(imagen: np.ndarray) -> np.ndarray:
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen_gris = cv2.resize(imagen_gris, (100, 100))
    lbp = local_binary_pattern(imagen_gris, N_PUNTOS, RADIO, METODO)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_PUNTOS + 3), range=(0, N_PUNTOS + 2))
    hist = hist.astype('float')
    hist /= hist.sum() + 1e-7
    return hist


def obtener_url_fuente_video(ruta: str, opciones_ydl: Optional[Dict[str, Any]] = None) -> str:
    if opciones_ydl is None:
        opciones_ydl = OPCIONES_YDL_POR_DEFECTO
    if ruta.startswith('http://') or ruta.startswith('https://'):
        with yt_dlp.YoutubeDL(opciones_ydl) as ydl:
            info = ydl.extract_info(ruta, download=False)
            return info.get('url', ruta)
    return ruta


def extraer_descriptores_faciales_desde_video(
    fuente: str,
    ruta_modelo: str,
    segundos_a_procesar: int = 100,
    saltar_frames: int = 5,
    opciones_ydl: Optional[Dict[str, Any]] = None,
    guardar_rostros: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    url_fuente = obtener_url_fuente_video(fuente, opciones_ydl)
    base_options = python.BaseOptions(model_asset_path=ruta_modelo)
    opciones = vision.FaceDetectorOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)

    descriptores: List[np.ndarray] = []
    rostros: List[np.ndarray] = []

    cap = cv2.VideoCapture(url_fuente)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(fps * segundos_a_procesar)
    indice_frame = 0

    with vision.FaceDetector.create_from_options(opciones) as detector:
        while cap.isOpened() and indice_frame < max_frames:
            exito, frame = cap.read()
            if not exito:
                break

            if indice_frame % saltar_frames == 0:
                timestamp_ms = int((indice_frame / fps) * 1000)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                resultado = detector.detect_for_video(imagen_mp, timestamp_ms)

                if resultado.detections:
                    for deteccion in resultado.detections:
                        caja = deteccion.bounding_box
                        x = int(caja.origin_x)
                        y = int(caja.origin_y)
                        ancho = int(caja.width)
                        alto = int(caja.height)
                        x, y = max(0, x), max(0, y)
                        recorte = frame[y:y + alto, x:x + ancho]

                        if recorte.size == 0:
                            continue

                        descriptores.append(obtener_descriptor_lbp(recorte))
                        if guardar_rostros:
                            rostros.append(recorte)

            indice_frame += 1

    cap.release()
    return descriptores, rostros


def agrupar_descriptores_optics(
    descriptores: List[np.ndarray],
    min_muestras: int = MIN_MUESTRAS_POR_DEFECTO,
    metrica: str = METRICA_POR_DEFECTO,
) -> np.ndarray:
    if not descriptores:
        return np.array([], dtype=int)

    X = np.vstack(descriptores)
    cluster = OPTICS(min_samples=min_muestras, metric=metrica).fit(X)
    return cluster.labels_


def guardar_rostros_agrupados(
    rostros: List[np.ndarray],
    etiquetas: np.ndarray,
    carpeta_salida: str,
) -> Dict[str, Any]:
    archivos_guardados: List[str] = []
    carpetas_clusters: List[str] = []

    os.makedirs(carpeta_salida, exist_ok=True)

    for i, etiqueta in enumerate(etiquetas):
        nombre_cluster = f'Persona_{etiqueta}' if etiqueta != -1 else 'Ruido'
        carpeta_cluster = os.path.join(carpeta_salida, nombre_cluster)
        os.makedirs(carpeta_cluster, exist_ok=True)
        ruta_archivo = os.path.join(carpeta_cluster, f'rostro_{i}.jpg')
        cv2.imwrite(ruta_archivo, rostros[i])
        archivos_guardados.append(ruta_archivo)
        if carpeta_cluster not in carpetas_clusters:
            carpetas_clusters.append(carpeta_cluster)

    return {'archivos_guardados': archivos_guardados, 'carpetas_clusters': carpetas_clusters}


def detectar_y_agrupar_rostros_video(
    fuente: str,
    ruta_modelo: str,
    carpeta_salida: str,
    segundos_a_procesar: int = 100,
    saltar_frames: int = 5,
    min_muestras: int = MIN_MUESTRAS_POR_DEFECTO,
    metrica: str = METRICA_POR_DEFECTO,
    opciones_ydl: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    descriptores, rostros = extraer_descriptores_faciales_desde_video(
        fuente,
        ruta_modelo,
        segundos_a_procesar=segundos_a_procesar,
        saltar_frames=saltar_frames,
        opciones_ydl=opciones_ydl,
        guardar_rostros=True,
    )
    etiquetas = agrupar_descriptores_optics(descriptores, min_muestras=min_muestras, metrica=metrica)
    info_guardada = guardar_rostros_agrupados(rostros, etiquetas, carpeta_salida) if rostros else {'archivos_guardados': [], 'carpetas_clusters': []}

    return {
        'descriptores': descriptores,
        'rostros': rostros,
        'etiquetas': etiquetas,
        'archivos_guardados': info_guardada['archivos_guardados'],
        'carpetas_clusters': info_guardada['carpetas_clusters'],
    }
