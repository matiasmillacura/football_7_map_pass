import gdown
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import supervision as sv
from ultralytics import YOLO
from configs.soccer import SoccerPitchConfiguration
from configs.view_transformer import ViewTransformer
from inference import get_model
import torch
import onnxruntime as ort
from decord import VideoReader, cpu, gpu

MODEL = "C:\\Users\\Matias\\Documents\\GitHub\\fotball_map_pass\\codes\\weights.onnx"
PLAYER_DETECTION_MODEL = YOLO(MODEL) # Configurar YOLO para utilizar la GPU

# Ruta del modelo
PITCH_DETECTION_MODEL_ID = "pitch-c3e9w/5"
PITCH_DETECTION_MODEL = get_model(PITCH_DETECTION_MODEL_ID, "49fMB8oQq6GxbnlRsfVd")

# Configuración de la cancha y tracker
CONFIG = SoccerPitchConfiguration()
tracker = sv.ByteTrack(
    track_activation_threshold=0.2,   # Umbral más bajo para asegurar que no se pierdan muchas detecciones válidas
    lost_track_buffer=60,             # Mayor buffer para soportar más tiempo de pérdida de detección (oclusiones o salidas de cámara)
    minimum_matching_threshold=0.9,   # Umbral de coincidencia alto para minimizar falsas coincidencias y evitar errores de reasignación de ID
    frame_rate=30,                    # Ajuste a la frecuencia de cuadros del video
    minimum_consecutive_frames=5      # Más cuadros para confirmar un seguimiento, lo que aumenta la estabilidad de la asignación de IDs
)

# Inicializar DataFrame para posiciones
posiciones_df = pd.DataFrame(columns=['Frame', 'Id', 'Pos X', 'Pos Y', 'Ball X', 'Ball Y'])

# URL de descarga de Google Drive y ruta de destino
VIDEO_URL = "https://drive.google.com/uc?id=1Ddyms2oDI359vt7eUhW48JVmbDg5Udsv"
VIDEO_PATH = "./GH033180.MP4"

# Descargar el video si no existe
if not os.path.exists(VIDEO_PATH):
    print("Descargando el video...")
    gdown.download(VIDEO_URL, VIDEO_PATH, quiet=False)

# Configurar el VideoReader de Decord para utilizar la GPU
vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
total_frames = len(vr)

# Leer el primer frame para la configuración de homografía
frame = vr[0].asnumpy()
result = PITCH_DETECTION_MODEL.infer(frame, confidence=0.9)[0]
key_points = sv.KeyPoints.from_inference(result)
filter = key_points.confidence[0] > 0.95
frame_reference_points = key_points.xy[0][filter]
pitch_reference_points = np.array(CONFIG.vertices)[filter]

view_transformer = ViewTransformer(
    source=pitch_reference_points,
    target=frame_reference_points
)

ultima_posicion_balon = None  # Inicialmente, no hay posición conocida del balón

# Procesar el video frame por frame
for frame_id in tqdm(range(total_frames)):
    frame = vr[frame_id].asnumpy()  # Obtener el frame como numpy array en GPU

    # Realizar la inferencia en GPU
    detections_result = PLAYER_DETECTION_MODEL.predict(frame, imgsz=1792, device='cpu')[0]
    boxes = detections_result.boxes.xyxy.cpu().numpy()
    scores = detections_result.boxes.conf.cpu().numpy()
    class_ids = detections_result.boxes.cls.cpu().numpy().astype(int)

    detections = sv.Detections(
        xyxy=boxes,
        confidence=scores,
        class_id=class_ids
    )

    # Filtrar el balón y jugadores
    ball_detections = detections[detections.class_id == 0]
    players_detections = detections[detections.class_id == 1]

    # Aplicar el tracker a las detecciones de jugadores
    all_detections = tracker.update_with_detections(detections=players_detections)

    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    frame_players_xy = all_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_positions = {id: xy for id, xy in zip(all_detections.tracker_id, frame_players_xy)}

    # Convertir frame_ball_xy a numpy array si tiene detecciones
    if frame_ball_xy is not None and len(frame_ball_xy) > 0:
        ultima_posicion_balon = frame_ball_xy[0]
        frame_ball_xy = np.array(frame_ball_xy)  # Convertir a numpy array
    elif ultima_posicion_balon is not None:
        frame_ball_xy = np.array([ultima_posicion_balon])  # Usar última posición conocida
    else:
        frame_ball_xy = np.array([])  # Array vacío si no hay posición previa

    # Transformar estas coordenadas a la cancha 2D
    if frame_ball_xy.size > 0:
        pitch_ball_xy = view_transformer.inverse_transform_points(frame_ball_xy)[0]
    else:
        pitch_ball_xy = None  # Si no hay posición, se asigna None para no registrar

    # Guardar las posiciones en el DataFrame si pitch_ball_xy tiene valores
    for player_id, player_position in players_positions.items():
        pitch_player_xy = view_transformer.inverse_transform_points(np.array([player_position]))[0]
        if pitch_ball_xy is not None:  # Solo registrar si pitch_ball_xy no es None
            posiciones_df = pd.concat([posiciones_df, pd.DataFrame([[frame_id, player_id, pitch_player_xy[0], pitch_player_xy[1], pitch_ball_xy[0], pitch_ball_xy[1]]],
                                                                   columns=posiciones_df.columns)], ignore_index=True)
        else:
            posiciones_df = pd.concat([posiciones_df, pd.DataFrame([[frame_id, player_id, pitch_player_xy[0], pitch_player_xy[1], None, None]],
                                                                   columns=posiciones_df.columns)], ignore_index=True)

# Guardar las posiciones en un archivo Excel
posiciones_df.to_excel('posiciones_jugadores_balon_nuevo.xlsx', index=False)

