import os
import pandas as pd
import requests
from yt_dlp import YoutubeDL
from urllib.parse import urlparse, parse_qs

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
    "https://www.youtube.com/watch?v=Chjtv8jlFy4",
    "https://www.youtube.com/watch?v=p5Z5o3Y2RGE",
    "https://www.youtube.com/watch?v=DhxCvCLyX28",
    "https://www.youtube.com/watch?v=tFN6K0P4I5o",
    "https://www.youtube.com/watch?v=_iAUVTo49X0",
    "https://www.youtube.com/watch?v=jQRqPAVqOXg",
    "https://www.youtube.com/watch?v=l8_-QlNKapk",
    "https://www.youtube.com/watch?v=e5BxviL18OU",
    "https://www.youtube.com/watch?v=zlHfc0w551s",
    "https://www.youtube.com/watch?v=f079K1f2WQk",
    "https://www.youtube.com/watch?v=pg827uDPFqA",
    "https://www.youtube.com/watch?v=QbfEqW1Ok-o",
    "https://www.youtube.com/watch?v=F_b1aMDliZs",
    "https://www.youtube.com/watch?v=4159g7iIzYs",
    "https://www.youtube.com/watch?v=FYKVVzCKIV8",
    "https://www.youtube.com/watch?v=iIXW5OFBbhA",
    "https://www.youtube.com/watch?v=IIPT2aUshoY",
    "https://www.youtube.com/watch?v=Tj-d2C4VoRQ",
    "https://www.youtube.com/watch?v=CHUGcDhhYho",
    "https://www.youtube.com/watch?v=tTnBdukIagk",
    "https://www.youtube.com/watch?v=OV2tKvUU9sM",
    "https://www.youtube.com/watch?v=ObpWslmNu3Y",
    "https://www.youtube.com/watch?v=RA9liM80Lj0",
    "https://www.youtube.com/watch?v=NGBLP4gaxck",
    "https://www.youtube.com/watch?v=fSQa9Rw5ciI",
    "https://www.youtube.com/watch?v=deDthQAegq8",
    "https://www.youtube.com/watch?v=pfG3VLb3HLo",
    "https://www.youtube.com/watch?v=UmP4sEgbh34",
    "https://www.youtube.com/watch?v=L9YkIfxd2r4",
    "https://www.youtube.com/watch?v=rHtg76J2hxE",
    "https://www.youtube.com/watch?v=rHEJDyn7Ksc",
    "https://www.youtube.com/watch?v=02Ifd-V9zmI",
    "https://www.youtube.com/watch?v=ec2mvL_ri80",
    "https://www.youtube.com/watch?v=R7GshNn2Ygc",
    "https://www.youtube.com/watch?v=1ls1t7OyHiQ",
    "https://www.youtube.com/watch?v=WOa9JGSnC-k",
    "https://www.youtube.com/watch?v=ZfnhfwdB7PQ",
    "https://www.youtube.com/watch?v=_NACkSeX8SE",
    "https://www.youtube.com/watch?v=qYjD_CG1LUs",
    "https://www.youtube.com/watch?v=LuZV9kkzscg",
    "https://www.youtube.com/watch?v=JF6EE8Raq1k",
    "https://www.youtube.com/watch?v=FriJDiVXH2c",
    "https://www.youtube.com/watch?v=ihTCa_K-WyY",
    "https://www.youtube.com/watch?v=6jRcm4GndeY",
    "https://www.youtube.com/watch?v=fJZpHmI26fU",
    "https://www.youtube.com/watch?v=VxP4MPYvxjA",
    "https://www.youtube.com/watch?v=gzhkg2lhloY"
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