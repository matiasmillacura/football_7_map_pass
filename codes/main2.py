import os
import numpy as np
import torchreid
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from configs.soccer import SoccerPitchConfiguration
import torch
from decord import VideoReader, cpu
from PIL import Image
from torchvision import transforms
import supervision as sv
from tqdm import tqdm
import supervision as sv
from ultralytics import YOLO
import numpy as np
from configs.view_transformer import ViewTransformer
import cv2
import torch
from configs.soccer import SoccerPitchConfiguration
from configs.drawing import draw_pitch, draw_points_on_pitch
import torchreid
from torchvision import transforms
import os
from configs.ball import BallAnnotator, BallTracker
# Configuración inicial
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Cargar modelos
reid_model = torchreid.models.build_model(name='resnet50', num_classes=12, pretrained=False)
checkpoint = torch.load('..\codes\models\model.pth.tar-300', map_location=device)
reid_model.load_state_dict(checkpoint['state_dict'])
reid_model = reid_model.to(device)
reid_model.eval()

PLAYER_DETECTION_MODEL = YOLO("..\codes\models\players.onnx")
BALL_DETECTION_MODEL = YOLO("..\codes\models\ball.onnx")

# Configuración de cancha y homografía
CONFIG = SoccerPitchConfiguration()
points_image = np.array([[891, 273], [905, 207], [1001, 217], [1057, 223], [1178, 240], [1232, 248], [957, 221], [1216, 256], [586, 518]], dtype=np.float32)
points_pitch = np.array([[2000, 1000], [0, 0], [0, 500], [0, 800], [0, 1200], [0, 1500], [500, 500], [500, 1500], [3500, 1500]], dtype=np.float32)
view_transformer = ViewTransformer(source=points_image, target=points_pitch)

# Configuración del video
VIDEO_PATH = "..\codes\videos\Pruebas.mov"
TARGET_VIDEO_PATH = "..\codes\videos\Pruebas_output.avi"
vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
total_frames = len(vr)

frame_width, frame_height = vr[0].shape[1], vr[0].shape[0]
fps = 59

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

# Configuración de preprocesamiento para reidentificación
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tracked_embeddings = {}
embedding_buffers = {}
player_tracks = {}
player_teams = {}
initial_positions = {}
buffer_size = 10
next_id = 1
team_decision_frames = 150

# Función para limpiar trayectoria del balón
def clean_ball_trajectory(ball_positions, distance_threshold=900):
    """Limpia posiciones inconsistentes del balón basadas en un umbral de distancia."""
    filtered_positions = []
    for i, pos in enumerate(ball_positions):
        if i == 0 or np.linalg.norm(np.array(pos) - np.array(filtered_positions[-1])) <= distance_threshold:
            filtered_positions.append(pos)
    return filtered_positions

# Función para limpiar posiciones de jugadores
def clean_player_positions(player_positions, distance_threshold=500):
    """Limpia posiciones inconsistentes de los jugadores basadas en un umbral de distancia."""
    filtered_positions = {}
    for player_id, positions in player_positions.items():
        filtered_positions[player_id] = []
        for i, pos in enumerate(positions):
            if i == 0 or np.linalg.norm(np.array(pos) - np.array(filtered_positions[player_id][-1])) <= distance_threshold:
                filtered_positions[player_id].append(pos)
    return filtered_positions

# Función para extraer embeddings
def extract_embedding(frame, bbox, model, preprocess, device):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = preprocess(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(crop).to(device).cpu().numpy()[0]
    return embedding

# Función para encontrar coincidencias en reidentificación
def find_best_match(embedding, tracked_embeddings, threshold=0.8):
    embedding = embedding / np.linalg.norm(embedding)
    best_match = None
    best_score = threshold
    for track_id, track_emb in tracked_embeddings.items():
        track_emb = track_emb / np.linalg.norm(track_emb)
        similarity = np.linalg.norm(embedding - track_emb)
        if similarity < best_score:
            best_match = track_id
            best_score = similarity
    return best_match

# Función para determinar equipo basado en posición
def determine_team(x_position):
    return "equipo_negro" if x_position < 1950 else "equipo_blanco"

# Función para dibujar caja y texto
def draw_player_box(frame, bbox, player_id, team_color):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, 2)
    cv2.putText(frame, f"ID: {player_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_color, 2)

def draw_box(frame, bbox, label, color=(255, 0, 0)):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# Configuración del slicer para mejorar la detección de balón
def callback(image_slice: np.ndarray) -> sv.Detections:
    result = BALL_DETECTION_MODEL(image_slice, imgsz=1024, conf=0.71)[0]
    return sv.Detections.from_ultralytics(result)

ball_tracker = BallTracker(buffer_size=20)


last_ball_positions = deque(maxlen=20)
player_position_buffers = {}
# Configuración del radar
radar_width = int(frame_width * 0.4)  # Ancho del radar: 40% del ancho del frame
radar_height = int(frame_height * 0.3)  # Alto del radar: 30% del alto del frame
radar_position = (int((frame_width - radar_width) / 2), int(frame_height - radar_height - 20))  # Posición centrada abajo



# Inicializar contador de frames
i = 0

# Procesamiento por frame
for i, frame in tqdm(enumerate(vr), total=total_frames):
    frame = frame.asnumpy()
    frame_ball = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- 1. Detección del balón ---
    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    detections_result_ball = slicer(frame_ball).with_nms(threshold=0.05)
    detections_result_ball = ball_tracker.update(detections_result_ball)
    detections_result_ball = detections_result_ball[detections_result_ball.class_id == 0]
    if len(detections_result_ball.xyxy) > 0:
        confidences = detections_result_ball.confidence
        best_idx = np.argmax(confidences)
        detections_result_ball.xyxy = detections_result_ball.xyxy[best_idx:best_idx+1]

    # Limpiar y proyectar posiciones del balón
    pitch_ball_positions, cleaned_positions = clean_ball_trajectory(
        detections_result_ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).tolist(), last_ball_positions
    )

    pitch_ball_positions = view_transformer.transform_points(np.array(pitch_ball_positions))
    last_ball_position = [pitch_ball_positions[0, 0], pitch_ball_positions[0, 1]] if len(pitch_ball_positions) > 0 else last_ball_position
    ball_x, ball_y = last_ball_position
    
    # --- 2. Detección de jugadores ---
    result_players = PLAYER_DETECTION_MODEL.predict(frame, imgsz=1792, iou=0.7)[0]
    detections_players = sv.Detections.from_ultralytics(result_players)
    players_detections = detections_players[detections_players.class_id == 1]
    frame_players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_xy = view_transformer.transform_points(frame_players_xy) if frame_players_xy is not None else []

    # Diccionario de posiciones de jugadores
    players_positions = {}
    detected_player_ids = []

    # --- 3. Reidentificación de jugadores ---
    for bbox, player_position, pitch_player_position in zip(players_detections.xyxy, frame_players_xy, pitch_players_xy):
        # Aplicar reidentificación únicamente a los jugadores
        embedding = extract_embedding(frame, bbox, reid_model, preprocess, device)
        if embedding is None:
            continue
        match_id = find_best_match(embedding, tracked_embeddings)
        if match_id is None:
            match_id = next_id
            next_id += 1
            tracked_embeddings[match_id] = embedding
        players_positions[match_id] = pitch_player_position
        detected_player_ids.append(match_id)

        if i < team_decision_frames:
            if match_id not in initial_positions:
                initial_positions[match_id] = []
            initial_positions[match_id].append(pitch_player_position[0])

    # Limpiar posiciones de jugadores
    cleaned_player_positions = clean_player_positions(player_position_buffers)

    # --- 4. Asignar equipos ---
    if i == team_decision_frames:
        for pid, positions in initial_positions.items():
            if positions:
                avg_x_position = np.mean(positions)
                player_teams[pid] = determine_team(avg_x_position)
    
        # --- 5. Dibujar cajas y texto según el equipo ---
    for bbox, match_id in zip(players_detections.xyxy, detected_player_ids):
        team_color = (0, 0, 0) if player_teams.get(match_id) == "equipo_negro" else (255, 255, 255)
        draw_player_box(frame, bbox, match_id, team_color)

    # --- 6. Dibujar la caja delimitadora del balón con confianza ---
    if len(detections_result_ball.xyxy) > 0:
        for bbox, confidence in zip(detections_result_ball.xyxy, detections_result_ball.confidence):
            label = f"Ball {confidence:.2f}"  # Etiqueta con confianza
            draw_box(frame, bbox, label, color=(0, 255, 255))  # Amarillo para el balón
        

    # --- 5. Generación de radar táctico ---
    radar = draw_pitch(CONFIG)
    pitch_players_black = np.array([pos for pid, pos_list in cleaned_player_positions.items() if player_teams.get(pid) == "equipo_negro" for pos in pos_list])
    pitch_players_white = np.array([pos for pid, pos_list in cleaned_player_positions.items() if player_teams.get(pid) == "equipo_blanco" for pos in pos_list])

    radar = draw_points_on_pitch(CONFIG, xy=pitch_players_black, face_color=sv.Color.BLACK, pitch=radar)
    radar = draw_points_on_pitch(CONFIG, xy=pitch_players_white, face_color=sv.Color.WHITE, pitch=radar)
    radar = draw_points_on_pitch(CONFIG, xy=pitch_ball_positions, face_color=sv.Color.BLUE, pitch=radar)

    # --- 6. Dibujar en el frame ---
    frame_with_radar = frame.copy()
    radar_resized = cv2.resize(radar, (int(frame_width * 0.4), int(frame_height * 0.3)))
    x, y = (int(frame_width * 0.3), int(frame_height * 0.7))
    alpha = 0.6
    for c in range(3):
        frame_with_radar[y:y+radar_resized.shape[0], x:x+radar_resized.shape[1], c] = (
            alpha * radar_resized[:, :, c] +
            (1 - alpha) * frame_with_radar[y:y+radar_resized.shape[0], x:x+radar_resized.shape[1], c]
        )

    # Escribir el frame en el video de salida
    out.write(frame_with_radar)

# Liberar recursos
out.release()
print("Procesamiento completado.")
