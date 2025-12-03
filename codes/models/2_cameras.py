import os
import cv2
import numpy as np
import torch
import torchreid
from tqdm import tqdm
from decord import VideoReader, cpu
from ultralytics import YOLO
from torchvision import transforms
from collections import deque

# Dispositivo (GPU si está disponible)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando dispositivo: {device}")

# Cargar el modelo de re-identificación
def load_reid_model(model_path):
    reid_model = torchreid.models.build_model(name='resnet50', num_classes=12, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    reid_model.load_state_dict(checkpoint['state_dict'])
    reid_model = reid_model.to(device)
    reid_model.eval()
    return reid_model

# Cargar el modelo de detección (jugadores y balón)
def load_detection_models(player_model_path, ball_model_path):
    player_model = YOLO(player_model_path)
    ball_model = YOLO(ball_model_path)
    return player_model, ball_model

# Preprocesar las imágenes para el modelo de re-identificación
def preprocess_image(crop, device):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    crop = preprocess(crop).unsqueeze(0).to(device)
    return crop

# Extraer el embedding de un jugador desde un frame
# Función para extraer embeddings (GPU)
def extract_embedding(frame, bbox, model, preprocess, device):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = preprocess(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(crop).to('cuda').cpu().numpy()[0]  # Procesar en GPU, devolver en CPU
    return embedding
# Encontrar la mejor coincidencia de un jugador usando su embedding
def find_best_match(embedding, tracked_embeddings, threshold=0.9):
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

# Dibujar las detecciones de jugadores en el frame
def draw_player_detections(frame, player_detections, tracked_embeddings, next_id, model, preprocess, device):
    for bbox in player_detections:
        # Extraer el embedding del jugador
        embedding = extract_embedding(frame, bbox, model, preprocess, device)
        if embedding is None:
            continue

        # Buscar la mejor coincidencia para el jugador
        match_id = find_best_match(embedding, tracked_embeddings)
        if match_id is None:
            match_id = next_id
            next_id += 1
            tracked_embeddings[match_id] = embedding

        # Extraer las coordenadas del jugador (bbox: [x1, y1, x2, y2])
        x1, y1, x2, y2 = map(int, bbox)
        player_position = [(x1 + x2) / 2, (y1 + y2) / 2]  # Centro del bbox

        # Dibujar las detecciones de los jugadores en el frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dibujar el bounding box
        cv2.putText(frame, f"ID: {match_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, next_id

# Dibujar las detecciones del balón en el frame
def draw_ball_detections(frame, ball_detections):
    for bbox in ball_detections:
        x1, y1, x2, y2 = map(int, bbox)
        ball_position = [(x1 + x2) / 2, (y1 + y2) / 2]  # Centro del bbox

        # Dibujar las detecciones del balón en el frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Dibujar el bounding box
        cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame


# Clase BallAnnotator: para visualizar la trayectoria del balón
class BallAnnotator:
    def __init__(self, max_positions=30):
        self.positions = deque(maxlen=max_positions)

    def annotate(self, frame, ball_position):
        self.positions.append(ball_position)

        # Dibujar las trayectorias anteriores del balón
        for i in range(1, len(self.positions)):
            cv2.line(frame, tuple(self.positions[i-1]), tuple(self.positions[i]), (0, 0, 255), 2)

        return frame

# Clase BallTracker: para rastrear el balón
class BallTracker:
    def __init__(self):
        self.positions = deque(maxlen=30)  # Guardar las últimas 30 posiciones

    def track(self, frame, ball_detections):
        if len(ball_detections) == 0:
            return frame, None

        # Obtener la posición del balón
        x1, y1, x2, y2 = map(int, ball_detections[0])
        ball_position = [(x1 + x2) / 2, (y1 + y2) / 2]
        self.positions.append(ball_position)

        # Dibujar las trayectorias del balón
        for i in range(1, len(self.positions)):
            cv2.line(frame, tuple(self.positions[i-1]), tuple(self.positions[i]), (0, 0, 255), 2)

        return frame, ball_position
    

# Procesar el video frame por frame
def process_video(video_path, player_model, ball_model, reid_model, output_video_path, device):
    vr = VideoReader(video_path, ctx=cpu(0))  # Usar GPU si está disponible
    total_frames = len(vr)

    # Obtener el primer frame para obtener las dimensiones
    frame = vr[0].asnumpy()
    height, width, _ = frame.shape

    # Crear el VideoWriter para guardar el video procesado
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))  # 30 FPS, tamaño igual al video original

    tracked_embeddings = {}  # Diccionario para los embeddings de jugadores
    next_id = 1  # El primer ID para los jugadores

    # Crear instancias de los anotadores y rastreadores
    ball_annotator = BallAnnotator(max_positions=30)
    ball_tracker = BallTracker()

    # Procesar los frames del video
    for frame_id in tqdm(range(total_frames)):
        frame = vr[frame_id].asnumpy()  # Obtener el frame como numpy array

        # Realizar detecciones en los jugadores
        player_detections_result = player_model.predict(frame, imgsz=1792, device=device)[0]
        player_boxes = player_detections_result.boxes.xyxy.cpu().numpy()
        player_scores = player_detections_result.boxes.conf.cpu().numpy()
        player_class_ids = player_detections_result.boxes.cls.cpu().numpy().astype(int)

        player_detections = []
        for box, score, class_id in zip(player_boxes, player_scores, player_class_ids):
            if class_id == 1 and score > 0.5:
                player_detections.append(box)

        # Realizar detecciones del balón
        ball_detections_result = ball_model.predict(frame, imgsz=1024, device=device)[0]
        ball_boxes = ball_detections_result.boxes.xyxy.cpu().numpy()
        ball_scores = ball_detections_result.boxes.conf.cpu().numpy()
        ball_class_ids = ball_detections_result.boxes.cls.cpu().numpy().astype(int)

        ball_detections = []
        for box, score, class_id in zip(ball_boxes, ball_scores, ball_class_ids):
            if class_id == 0 and score > 0.5:
                ball_detections.append(box)

        # Dibujar las detecciones de jugadores
        frame, next_id = draw_player_detections(frame, player_detections, tracked_embeddings, next_id, reid_model, preprocess_image, device)

        # Dibujar las detecciones del balón
        frame = draw_ball_detections(frame, ball_detections)

        # Anotar la trayectoria del balón
        if ball_detections:
            ball_position = ball_tracker.track(frame, ball_detections)[1]
            frame = ball_annotator.annotate(frame, ball_position)

        # Guardar el frame procesado
        out.write(frame)

    # Liberar recursos
    out.release()
    print(f'Video procesado guardado en: {output_video_path}')

# Función principal para ejecutar el flujo completo
# Función principal para ejecutar el flujo completo
# Función principal para ejecutar el flujo completo
def run_detection_and_tracking(video_path, player_model_path, ball_model_path, reid_model_path, output_video_path, device):
    # Cargar los modelos
    player_model, ball_model = load_detection_models(player_model_path, ball_model_path)
    reid_model = load_reid_model(reid_model_path)
    # Procesar el video

    process_video(video_path, player_model, ball_model, reid_model, output_video_path)

# Llamada a la función principal (ajustar las rutas según sea necesario)
video_path = "/home/mmillacura/intento1/fotball_map_pass/codes/Esquina2.mov"
player_model_path = "/home/mmillacura/intento1/fotball_map_pass/codes/weights.onnx"
ball_model_path = "/home/mmillacura/sports/models_cache/ball-detect-w4l13/1/weights.onnx"
reid_model_path = "/home/mmillacura/intento1/fotball_map_pass/codes/log/resnet50/model/model.pth.tar-300"
output_video_path = "/home/mmillacura/sports/output.mp4"

run_detection_and_tracking(video_path, player_model_path, ball_model_path, reid_model_path, output_video_path)

