import cv2
import requests
import tempfile
import os
from models.yolo import YOLO
from utils.video_processing import process_video

# URLs de los archivos
cfg_url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'
weights_url = 'https://pjreddie.com/media/files/yolov3.weights'
names_url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'

# Función para descargar un archivo desde una URL y guardarlo en un archivo temporal
def download_file(url, suffix):
    response = requests.get(url)
    response.raise_for_status()  # Asegura que la descarga fue exitosa
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(response.content)
    temp_file.close()
    return temp_file.name

def main():
    # Descargar los archivos y guardarlos en archivos temporales
    cfg_path = download_file(cfg_url, '.cfg')
    weights_path = download_file(weights_url, '.weights')
    names_path = download_file(names_url, '.names')

    # Depuración: imprimir rutas de archivos
    print(f"Config path: {cfg_path}")
    print(f"Weights path: {weights_path}")
    print(f"Names path: {names_path}")

    # Verificar si los archivos fueron descargados correctamente
    if not (cfg_path and weights_path and names_path):
        print("Failed to download the necessary files.")
        return

    # Inicializar el modelo YOLO con las rutas temporales
    yolo = YOLO(
        model_path=weights_path,
        config_path=cfg_path,
        classes_path=names_path
    )

    # Ruta del video a procesar
    video_path = "/Users/victormedina/UNI/Developer/Football_Analytics/videos/Villarreal-vs-Real-Madrid-4-4-highlights-SPORTDAYLIGHT.COM_.mp4"
    
    # Procesar el video con el modelo YOLO
    process_video(video_path, yolo)

    # Eliminar archivos temporales
    os.remove(cfg_path)
    os.remove(weights_path)
    os.remove(names_path)

if __name__ == "__main__":
    main()
