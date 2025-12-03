from configs.ball import BallTracker, BallAnnotator
import supervision as sv
import os
import cv2
import numpy as np
import torch
import torchreid
from tqdm import tqdm
from ultralytics import YOLO
from torchvision import transforms
from collections import deque
from matplotlib import cm
from configs.team import TeamClassifier  # Clase importada
from configs.soccer import SoccerPitchConfiguration
from configs.drawing import draw_pitch, draw_points_on_pitch


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


# Función para extraer embedding de un crop
def extract_embedding(frame: np.ndarray, det: np.ndarray, reid_model, device):
    """
    Extrae el embedding de un jugador a partir de su crop usando el modelo de re-identificación.
    """
    # Extraemos el crop de la detección
    crop = sv.crop_image(frame, det)
    
    # Preprocesamos la imagen para pasarla al modelo de re-id
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    crop = preprocess(crop).unsqueeze(0).to(device)  # Añadimos batch dimension
    
    # Realizamos la inferencia
    with torch.no_grad():
        embedding = reid_model(crop).cpu().numpy()
    
    return embedding

# Función para encontrar la mejor coincidencia entre embeddings
def find_best_match(embedding, tracked_embeddings, threshold=0.5):
    """
    Encuentra la mejor coincidencia para un embedding dado comparándolo con los embeddings almacenados.
    Si la coincidencia es lo suficientemente cercana, devuelve el ID correspondiente.
    """
    best_match_id = None
    best_distance = float('inf')

    for track_id, stored_embedding in tracked_embeddings.items():
        # Calculamos la distancia euclidiana entre el embedding nuevo y el almacenado
        distance = np.linalg.norm(embedding - stored_embedding)

        # Si la distancia es menor que el umbral, lo consideramos una coincidencia
        if distance < best_distance and distance < threshold:
            best_distance = distance
            best_match_id = track_id

    return best_match_id

import numpy.typing as npt


# Clase para la transformación de vista
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m, _ = cv2.findHomography(source, target)
        self.m_inv = np.linalg.inv(self.m)  # Calcular la matriz inversa para la transformación inversa

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        points = points.reshape(-1, 1, 2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.m)
        return points.reshape(-1, 2).astype(np.float32)
    
    def inverse_transform_points(self, points: np.ndarray) -> np.ndarray:
        points = points.reshape(-1, 1, 2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.m_inv)
        return points.reshape(-1, 2).astype(np.float32)

def process_image(image_path, player_model, homography_matrix, device):
    # Leer la imagen
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"No se pudo leer la imagen en {image_path}")
        return

    # Crear configuración de cancha
    pitch_config = SoccerPitchConfiguration()

    # Detectar jugadores
    result = player_model.predict(frame, imgsz=1792, device=device)[0]
    detections = sv.Detections.from_ultralytics(result)
    player_detections = detections[detections.class_id == 1]

    # Validar detecciones
    if len(player_detections) == 0:
        print("No se detectaron jugadores en esta imagen.")
        return

    # Preparar radar táctico
    radar_pitch = draw_pitch(pitch_config)

    # Dibujar jugadores con IDs
    projected_players = []
    player_labels = []

    # Extraer las coordenadas de las cajas delimitadoras de los jugadores
    bboxes_p_c = player_detections.xyxy  # Coordenadas de las cajas (x1, y1, x2, y2)
    
    # Calculamos la posición central de cada caja (promedio entre las coordenadas de las esquinas)
    detected_ppos_src_pts = bboxes_p_c[:, :2] + np.array([[0] * bboxes_p_c.shape[0], bboxes_p_c[:, 3] / 2]).transpose()

    # Inicializar la clase ViewTransformer con la matriz de homografía
    transformer = ViewTransformer(homography_matrix)

    # Transformar las coordenadas de los jugadores al mapa táctico
    transformed_pts = transformer.inverse_transform_points(detected_ppos_src_pts)

    # Añadir los puntos transformados al radar táctico
    for i, projected_point in enumerate(transformed_pts):
        player_labels.append(f"Jugador {i + 1}")
        projected_players.append(projected_point)

    # Proyectar jugadores en el radar táctico
    print("Proyectando jugadores en el radar táctico...")

    radar_pitch = draw_points_on_pitch(
        config=pitch_config,
        xy=transformed_pts,  # Puntos proyectados de jugadores
        face_color=sv.Color.GREEN,  # Color de los puntos
        edge_color=sv.Color.BLACK,  # Color del borde de los puntos
        pitch=radar_pitch  # Imagen de la cancha
    )
    
    # Guardar la imagen procesada
    output_image_path = "output.jpg"
    cv2.imwrite(output_image_path, frame)
    radar_image_path = "output_radar.jpg"
    cv2.imwrite(radar_image_path, radar_pitch)
    
    print(f"Imagen procesada guardada en {output_image_path}")
    print(f"Radar táctico guardado en {radar_image_path}")




# Función principal
def run_image_processing(image_path, player_model_path, ball_model_path, reid_model_path, output_image_path, homography_matrix):
    player_model, ball_model = load_detection_models(player_model_path, ball_model_path)
    reid_model = load_reid_model(reid_model_path)
    process_image(image_path, player_model, homography_matrix, device)

homography_matrix = np.array([
    [ 1.11271953e+00, -9.09806366e-01,  1.60369326e+03],
    [ 1.17716476e+00,  1.21698133e+01,  -3.19022566e+03],
    [ -3.24412230e-03, 3.79757845e-02,  1.00000000e+00]
])


# Rutas
image_path = "C:\\Users\\Matias\\Documents\\GitHub\\fotball_map_pass\\codes\\es2.jpg"
player_model_path = "C:\\Users\\Matias\\Documents\\GitHub\\fotball_map_pass\\models\\weights.onnx"
ball_model_path = "C:\\Users\\Matias\\Documents\\GitHub\\fotball_map_pass\\models\\ball.onnx"
reid_model_path = "C:\\Users\\Matias\\Documents\\GitHub\\fotball_map_pass\\models\\model.pth.tar-300"
output_image_path = "C:\\Users\\Matias\\Documents\\GitHub\\fotball_map_pass\\codes\\output.jpg"

# Ejecutar
run_image_processing(image_path, player_model_path, ball_model_path, reid_model_path, output_image_path, homography_matrix)