from multiprocessing import Process, Queue
from tqdm import tqdm
import pandas as pd
import numpy as np
from supervision import Detections
from inicializar import *  # Importa los modelos inicializados
import torch

def process_video(video_path, queue, process_id):
    """
    Procesa un video para detección de jugadores y reidentificación.
    Los resultados se envían a una cola para combinarse después.
    """
    # Configurar lector de video
    from decord import VideoReader, cpu

    
    video_reader = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(video_reader)

    # DataFrame para almacenar resultados
    posiciones_df = pd.DataFrame(columns=['Frame', 'Id', 'Pos X', 'Pos Y', 'Ball X', 'Ball Y'])

    # Variables locales para seguimiento
    tracked_embeddings = {}
    next_id = 1

    # Procesar cada frame
    for frame_id in tqdm(range(total_frames), desc=f"Procesando Video {process_id}"):
        frame = video_reader[frame_id].asnumpy()

        # Detección de jugadores y balón
        detections_result = PLAYER_DETECTION_MODEL.predict(frame, imgsz=1792, device=device)[0]
        boxes = detections_result.boxes.xyxy.cpu().numpy()
        scores = detections_result.boxes.conf.cpu().numpy()
        class_ids = detections_result.boxes.cls.cpu().numpy().astype(int)

        detections = Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=class_ids
        )

        # Filtrar detecciones
        ball_detections = detections[detections.class_id == 0]
        players_detections = detections[detections.class_id == 1]

        # ReID para jugadores
        players_positions = {}
        for bbox in players_detections.xyxy:
            embedding = extract_embedding(frame, bbox, reid_model, preprocess, device)
            if embedding is None:
                continue
            match_id = find_best_match(embedding, tracked_embeddings)
            if match_id is None:
                match_id = next_id
                next_id += 1
                tracked_embeddings[match_id] = embedding
            # Posición ficticia (puedes integrar homografía aquí si es necesario)
            players_positions[match_id] = np.mean(bbox[:2] + bbox[2:], axis=0)

        # Manejar posición del balón
        ball_position = None
        if len(ball_detections.xyxy) > 0:
            ball_position = np.mean(ball_detections.xyxy[0][:2] + ball_detections.xyxy[0][2:], axis=0)

        # Registrar datos en el DataFrame
        # Registrar datos en el DataFrame
        for player_id, player_pos in players_positions.items():
            # Validar que player_pos tenga dos elementos
            if player_pos is None or len(player_pos) != 2:
                print(f"Advertencia: Posición inválida para el jugador {player_id} en el frame {frame_id}.")
                continue
            
            # Validar ball_position
            ball_position_x = ball_position[0] if ball_position is not None else None
            ball_position_y = ball_position[1] if ball_position is not None else None
        
            # Agregar datos al DataFrame
            posiciones_df = pd.concat([posiciones_df, pd.DataFrame([[
                frame_id,
                player_id,
                player_pos[0], player_pos[1],
                ball_position_x,
                ball_position_y
            ]], columns=posiciones_df.columns)], ignore_index=True)
        
        
            # Enviar resultados a la cola
            queue.put((process_id, posiciones_df))

def extract_embedding(frame, bbox, model, preprocess, device):
    """
    Extrae un embedding para una caja delimitadora dada.
    """
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = preprocess(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(crop).cpu().numpy()[0]
    return embedding

def find_best_match(embedding, tracked_embeddings, threshold=0.88):
    """
    Encuentra el mejor match para un embedding en los embeddings ya rastreados.
    """
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