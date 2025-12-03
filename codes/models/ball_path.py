from collections import deque
import supervision as sv
from configs.drawing import draw_pitch, draw_points_on_pitch, draw_paths_on_pitch
from configs.soccer import SoccerPitchConfiguration
from tqdm import tqdm
from ultralytics import YOLO
from configs.view_transformer import ViewTransformer
import numpy as np
from inference import get_model

'''
Este codigo, describe la trayectoria de la pelota con respecto a las detecciones de esta


'''


SOURCE_VIDEO_PATH =  "..\\fotball_map_pass\\videos\\video2.mp4"
# Ruta del modelo y del video
PLAYER_DETECTION_MODEL = YOLO("..\\fotball_map_pass\\models\\weights.onnx")

# Configuración del modelo de detección de cancha
PITCH_DETECTION_MODEL_ID = "pitch-c3e9w/5"
PITCH_DETECTION_MODEL = get_model(PITCH_DETECTION_MODEL_ID, "49fMB8oQq6GxbnlRsfVd")

BALL_ID = 0
MAXLEN = 5
# Configuración de la cancha virtualizada
CONFIG = SoccerPitchConfiguration()


video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

path_raw = []
M = deque(maxlen=MAXLEN)

for frame in tqdm(frame_generator, total=video_info.total_frames):

    # Realizar la inferencia
    result = PLAYER_DETECTION_MODEL.predict(frame, imgsz=1792)[0]

    # Extraer las detecciones manualmente
    boxes = result.boxes.xyxy.cpu().numpy()  # Coordenadas de las cajas
    scores = result.boxes.conf.cpu().numpy()  # Puntajes de confianza
    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # IDs de clases

    # Convertir las detecciones a un formato compatible con Supervision
    detections = sv.Detections(
        xyxy=boxes,
        confidence=scores,
        class_id=class_ids
    )

    ball_detections = detections[detections.class_id == BALL_ID]
    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    result = PITCH_DETECTION_MODEL.infer(frame, confidence=0.9)[0]
    key_points = sv.KeyPoints.from_inference(result)

    filter = key_points.confidence[0] > 0.9
    frame_reference_points = key_points.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]

    transformer = ViewTransformer(
        source=frame_reference_points,
        target=pitch_reference_points
    )
    M.append(transformer.m)
    transformer.m = np.mean(np.array(M), axis=0)

    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    # Verificar si hay detecciones válidas antes de transformar
    if frame_ball_xy is not None and len(frame_ball_xy) > 0:
        pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)
        path_raw.append(pitch_ball_xy)
    else:
        path_raw.append(np.empty((0, 2), dtype=np.float32))  # Añadir una estructura vacía si no hay detecciones

path = [
    np.empty((0, 2), dtype=np.float32) if coorinates.shape[0] >= 2 else coorinates
    for coorinates
    in path_raw
]

path = [coorinates.flatten() for coorinates in path]


annotated_frame = draw_pitch(CONFIG)
annotated_frame = draw_paths_on_pitch(
    config=CONFIG,
    paths=[path],
    color=sv.Color.WHITE,
    pitch=annotated_frame)

sv.plot_image(annotated_frame)
