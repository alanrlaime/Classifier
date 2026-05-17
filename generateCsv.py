import os
import pandas as pd
import requests
from yt_dlp import YoutubeDL
from urllib.parse import urlparse, parse_qs

# =========================================
# CONFIG
# =========================================


# Lista de URLs de YouTube
video_urls = [
    "https://www.youtube.com/watch?v=jkREZ1KHgQo",
    "https://www.youtube.com/watch?v=IzyjztxOIoY",
    "https://www.youtube.com/watch?v=5lHYykk7suA",
    "https://www.youtube.com/watch?v=4i4p_S2MPU0",
    "https://www.youtube.com/watch?v=7v_OLMDKFFg",
    "https://www.youtube.com/watch?v=n6XvfUG4QBQ",
    "https://www.youtube.com/watch?v=CbZPHgN6e90",
    "https://www.youtube.com/watch?v=fGHv_OlPKjw",
    "https://www.youtube.com/watch?v=kvLonxrJvtU",
    "https://www.youtube.com/watch?v=TvqJQukWk8o",
    "https://www.youtube.com/watch?v=vVxPVJ3FGnM",
    "https://www.youtube.com/watch?v=rJNcf_cEunA",
    "https://www.youtube.com/watch?v=Mrmny5d0Ik4",
    "https://www.youtube.com/watch?v=DwAMhJFnRsM",
    "https://www.youtube.com/watch?v=zdGLfb4L5SM",
    "https://www.youtube.com/watch?v=PkDHVDeP_B4",
    "https://www.youtube.com/watch?v=cMkl8-3IArM",
    "https://www.youtube.com/watch?v=2vxoMxe-uvM",
    "https://www.youtube.com/watch?v=aZT8kyTlxUo",
    "https://www.youtube.com/watch?v=md36rwkuG20",
    "https://www.youtube.com/watch?v=Ok7QV07OZD4",
    "https://www.youtube.com/watch?v=9BRworAAbkc",
    "https://www.youtube.com/watch?v=2XPNfkJA4x8",
    "https://www.youtube.com/watch?v=-GLlauTt8eI",
    "https://www.youtube.com/watch?v=qNY-WAm64Ks",
    "https://www.youtube.com/watch?v=Upr_HxRPfe4",
    "https://www.youtube.com/watch?v=vCvL5cvCzOg",
    "https://www.youtube.com/watch?v=Chjtv8jlFy4"
]


PORTADAS_DIR = "portadas-yt"
CSV_OUTPUT = "dataset_youtube.csv"

os.makedirs(PORTADAS_DIR, exist_ok=True)

# =========================================
# DESCARGAR PORTADA
# =========================================

def descargar_portada(url, filename):

    if not url:
        return None

    try:
        response = requests.get(url, timeout=15)

        if response.status_code == 200:

            path = os.path.join(PORTADAS_DIR, filename)

            with open(path, "wb") as f:
                f.write(response.content)

            return path

    except Exception as e:
        print("Error portada:", e)

    return None

# =========================================
# VIDEO ID
# =========================================

def obtener_video_id(url):

    parsed = urlparse(url)

    if "youtube.com" in parsed.netloc:
        return parse_qs(parsed.query).get("v", [None])[0]

    elif "youtu.be" in parsed.netloc:
        return parsed.path.strip("/")

    return None

# =========================================
# YDL CONFIG
# =========================================

ydl_opts = {
    "quiet": True,
    "skip_download": True,
    "noplaylist": True,
    "extract_flat": False,
}

dataset = []

# =========================================
# EXTRAER DATOS
# =========================================

with YoutubeDL(ydl_opts) as ydl:

    for idx, url in enumerate(video_urls, start=1):

        try:

            info = ydl.extract_info(url, download=False)

            formatos = info.get("formats", [])

            # --------------------------------
            # FILTRAR FORMATOS VALIDOS
            # --------------------------------

            formatos_validos = [
                f for f in formatos
                if f.get("height") is not None
            ]

            mejor_formato = None

            if formatos_validos:

                mejor_formato = max(
                    formatos_validos,
                    key=lambda x: x.get("height", 0)
                )

            # --------------------------------
            # DATOS
            # --------------------------------

            titulo = info.get("title")
            canal = info.get("uploader")
            fecha = info.get("upload_date")
            duracion = info.get("duration")
            vistas = info.get("view_count")
            likes = info.get("like_count")

            descripcion = info.get("description")

            resolucion = None
            fps = None
            extension = None
            filesize = None

            if mejor_formato:

                resolucion = (
                    f"{mejor_formato.get('width')}x"
                    f"{mejor_formato.get('height')}"
                )

                fps = mejor_formato.get("fps")
                extension = mejor_formato.get("ext")
                filesize = mejor_formato.get("filesize")

            # --------------------------------
            # THUMBNAIL
            # --------------------------------

            thumbnail = info.get("thumbnail")

            video_id = obtener_video_id(url)

            portada_path = descargar_portada(
                thumbnail,
                f"{video_id}.jpg"
            )

            # --------------------------------
            # GUARDAR
            # --------------------------------

            dataset.append({

                "ID": idx,
                "URL": url,
                "Video_ID": video_id,
                "Titulo": titulo,
                "Canal": canal,
                "Fecha_Subida": fecha,
                "Duracion_Segundos": duracion,
                "Resolucion": resolucion,
                "FPS": fps,
                "Formato": extension,
                "FileSize_Bytes": filesize,
                "Thumbnail_URL": thumbnail,
                "Ruta_Portada": portada_path,
            })

            print(f"[OK] {titulo}")

        except Exception as e:

            print(f"[ERROR] {url}")
            print(e)

# =========================================
# CSV
# =========================================

df = pd.DataFrame(dataset)

df.to_csv(
    CSV_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("\nCSV generado:", CSV_OUTPUT)