import os
import numpy as np
import torchreid
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from configs.soccer import SoccerPitchConfiguration
from configs.view_transformer import ViewTransformer
import torch
from decord import VideoReader, cpu
from PIL import Image
from torchvision import transforms
import supervision as sv
from tqdm import tqdm
import supervision as sv
from ultralytics import YOLO
import numpy as np
import cv2
import torch
from configs.soccer import SoccerPitchConfiguration
from configs.drawing import draw_pitch, draw_points_on_pitch
from configs.view_transformer import ViewTransformer
import torchreid
from torchvision import transforms
import os
from configs.ball import BallAnnotator, BallTracker

# Ruta absoluta al modelo
BALL_DETECTION_MODEL = YOLO("C:\\Users\\Matias\\Documents\\GitHub\\fotball_map_pass\\models\\ball.onnx")
VIDEO_PATH = "C:\\Users\\Matias\\Documents\\GitHub\\fotball_map_pass\\videos\\Esquina2.mov"

# Configuración de Decord para leer el video
vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
total_frames = len(vr)

# Inicializar el rastreador y el anotador del balón
ball_tracker = BallTracker(buffer_size=20)
ball_annotator = BallAnnotator(radius=6, buffer_size=10)

# Configuración del slicer para mejorar la detección del balón
def callback(image_slice: np.ndarray) -> sv.Detections:
    result = BALL_DETECTION_MODEL(image_slice, imgsz=1024)[0]
    return sv.Detections.from_ultralytics(result)
# Configuración de la cancha y homografía
CONFIG = SoccerPitchConfiguration()


for i, frame in tqdm(enumerate(vr), total=total_frames):
    frame = frame.asnumpy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    radar = draw_pitch(config=CONFIG)

    # Inferencia del balón utilizando el slicer
    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )
    detections_result_ball = slicer(frame).with_nms(threshold=0.05)
    detections_result_ball = ball_tracker.update(detections_result_ball)

    # Filtrar solo detecciones de balón
    ball_detections = detections_result_ball[detections_result_ball.class_id == 0]

    # Acceder a las coordenadas de cada detección
    for box in ball_detections.xyxy:
        x_min, y_min, x_max, y_max = box
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        print(f"Ball Position: x_center={x_center}, y_center={y_center}")

    # Anotación en el frame
    annotated_frame = frame.copy()
    annotated_frame = ball_annotator.annotate(annotated_frame, ball_detections)

    # Mostrar el frame procesado para depuración
    cv2.imshow("Video con Radar", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
