import numpy as np
import torch
from ultralytics import YOLO
from decord import VideoReader, cpu
import cv2
from configs.soccer import SoccerPitchConfiguration
from configs.drawing import draw_pitch, draw_points_on_pitch
import supervision as sv  # Asume que se usa la librería Supervision

# Clase para manejar transformaciones
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m, _ = cv2.findHomography(source, target)
        self.m_inv = np.linalg.inv(self.m)  # Matriz inversa para transformación inversa

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        points = points.reshape(-1, 1, 2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.m)
        return points.reshape(-1, 2).astype(np.float32)
    
    def inverse_transform_points(self, points: np.ndarray) -> np.ndarray:
        points = points.reshape(-1, 1, 2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.m_inv)
        return points.reshape(-1, 2).astype(np.float32)

# Configuración del dispositivo
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Modelo de detección (YOLO)
MODEL = "C:\\Users\\Matias\\Documents\\GitHub\\fotball_map_pass\\models\\weights.onnx"
PLAYER_DETECTION_MODEL = YOLO(MODEL)

# Video
VIDEO_PATH = "C:\\Users\\Matias\\Documents\\GitHub\\fotball_map_pass\\videos\\Esquina1.mov"
vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
total_frames = len(vr)

# Configuración de la cancha
pitch_config = SoccerPitchConfiguration()

# Puntos de referencia para calcular la homografía
camera_points = np.array([
    [1347, 371], [1126, 370], [1064, 414], [450, 506], [813, 595]
], dtype=np.float32)

# Puntos correspondientes en el plano 2D usando los vértices de pitch_config
field_points = np.array([
    pitch_config.vertices[16],  # Vertice 17 -> [3500, 1500]
    pitch_config.vertices[15],  # Vertice 16 -> [3500, 500]
    pitch_config.vertices[0],   # Vertice 1 -> [2000, 1000]
    pitch_config.vertices[7],   # Vertice 8 -> [500, 500]
    pitch_config.vertices[8],   # Vertice 9 -> [500, 1500]
], dtype=np.float32)

# Crear el transformador de vista
view_transformer = ViewTransformer(camera_points, field_points)

# Procesar frames
for frame_id in range(total_frames):
    # Redibujar la cancha base en cada iteración
    field_image = draw_pitch(config=pitch_config, scale=0.1, padding=50)

    # Leer el frame del video
    frame = vr[frame_id].asnumpy()

    # Detectar jugadores y balón en el video
    results = PLAYER_DETECTION_MODEL.predict(frame, imgsz=1792, device=device)[0]
    detections = sv.Detections(
        xyxy=results.boxes.xyxy.cpu().numpy(),
        confidence=results.boxes.conf.cpu().numpy(),
        class_id=results.boxes.cls.cpu().numpy().astype(int)
    )

    # Filtrar detecciones de balón y jugadores
    ball_detections = detections[detections.class_id == 0]  # ID 0: balón
    players_detections = detections[detections.class_id == 1]  # ID 1: jugadores

    # Obtener coordenadas de los centros
    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    frame_players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    # Transformar coordenadas al plano 2D
    pitch_ball_xy = view_transformer.inverse_transform_points(frame_ball_xy) if len(frame_ball_xy) > 0 else []
    pitch_players_xy = view_transformer.inverse_transform_points(frame_players_xy) if len(frame_players_xy) > 0 else []

    # Dibujar las posiciones en la cancha 2D
    if len(pitch_players_xy) > 0:
        field_image = draw_points_on_pitch(
            config=pitch_config,
            xy=pitch_players_xy,
            face_color=(0, 0, 255),  # Azul para jugadores
            pitch=field_image
        )
    if len(pitch_ball_xy) > 0:
        field_image = draw_points_on_pitch(
            config=pitch_config,
            xy=pitch_ball_xy,
            face_color=(0, 255, 0),  # Verde para balón
            pitch=field_image
        )

    # Mostrar resultados
    cv2.imshow("Cancha 2D con Detecciones", field_image)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
