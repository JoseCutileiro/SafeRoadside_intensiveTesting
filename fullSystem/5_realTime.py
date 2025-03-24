import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque
import random
import pyautogui
import logging
import sys
import math
import struct
import time

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DE LOGGING
# -----------------------------------------------------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger("ultralytics").setLevel(logging.WARNING)


# -----------------------------------------------------------------------------
# 0) LER O ARQUIVO map.txt
# -----------------------------------------------------------------------------
def load_map_data(map_file="map.txt"):
    """
    Lê lat0_deg, lon0_deg e a matriz de homografia de um arquivo map.txt
    no formato:
       lat0 lon0
       h11 h12 h13
       h21 h22 h23
       h31 h32 h33
    """
    try:
        with open(map_file, "r") as f:
            lines = f.readlines()

        # Primeira linha: lat0_deg, lon0_deg
        lat0_deg, lon0_deg = map(float, lines[0].split())

        # Próximas 3 linhas: matriz de homografia
        h_rows = []
        for i in range(1, 4):
            row_vals = list(map(float, lines[i].split()))
            h_rows.append(row_vals)

        homography_mat = np.array(h_rows, dtype=np.float32)

        if homography_mat.shape != (3, 3):
            raise ValueError("A matriz de homografia não possui dimensões 3x3.")

        return lat0_deg, lon0_deg, homography_mat

    except Exception as e:
        print(f"Erro ao ler {map_file}: {e}")
        sys.exit(1)


# -----------------------------------------------------------------------------
# FUNÇÃO P/ CONVERTER (X, Y) EM LAT/LON
# -----------------------------------------------------------------------------
def xy_to_latlon(X, Y, lat0_deg, lon0_deg):
    """
    Inverso da projeção local.
    Dado (X, Y) em metros no sistema local,
    devolve (lat, lon) em graus decimais.
    """
    R = 6371000.0  # raio aproximado da Terra em metros
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)

    lat = Y / R + lat0
    lon = X / (R * math.cos(lat0)) + lon0

    # Converter p/ graus
    lat_deg = math.degrees(lat)
    lon_deg = math.degrees(lon)
    return lat_deg, lon_deg


# -----------------------------------------------------------------------------
# 1) CARREGAR TRAJETÓRIAS PREDEFINIDAS
# -----------------------------------------------------------------------------
def load_trajectories(file_path):
    """Carrega e retorna uma lista de trajetórias pré-definidas."""
    trajectories = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("Traj: "):
            points_str = line[len("Traj: "):].split('), (')
            points = []
            for p in points_str:
                p = p.replace('(', '').replace(')', '')
                x, y = map(int, p.split(', '))
                points.append((x, y))
            trajectories.append(np.array(points, dtype=np.int32))

    return trajectories


# -----------------------------------------------------------------------------
# 2) FUNÇÃO PARA ENCONTRAR A MELHOR TRAJETÓRIA (VEÍCULOS)
# -----------------------------------------------------------------------------
def find_best_trajectory(current_traj, predefined_trajs, max_points=50, score_threshold=3000):
    """Encontra a melhor continuação de trajetória com base em uma lista de trajetórias pré-definidas."""
    current_length = len(current_traj)
    if current_length < 4:
        return []

    current_arr = np.array(current_traj, dtype=np.float32)
    best_score = float('inf')
    best_match = []

    for traj in predefined_trajs:
        # Só consideramos trajetórias pré-definidas maiores do que a atual
        if len(traj) <= current_length:
            continue

        # Criamos uma janela deslizante para comparar com a current_traj
        for i in range(len(traj) - current_length):
            reference_window = traj[i : i + current_length]
            # Soma das distâncias entre pontos equivalentes
            diff = current_arr - reference_window
            score = np.sum(np.linalg.norm(diff, axis=1))

            if score < best_score:
                best_score = score
                # Pega próximos pontos para prever
                start_pred = i + current_length
                end_pred = start_pred + max_points
                best_match = traj[start_pred:end_pred]

    # Retorna somente se o score encontrado for significativo
    if best_score < score_threshold:
        return best_match
    else:
        return []


# -----------------------------------------------------------------------------
# 3) FILTRO DE KALMAN ESTENDIDO (EKF) PARA PEDESTRES
# -----------------------------------------------------------------------------
def ekf(previous_points, prediction_range=5):
    """
    Retorna uma lista de pontos previstos a partir dos 'previous_points'
    usando um EKF simplificado.
    """
    if len(previous_points) < 3:
        return []

    dt = 1.0
    Q = np.eye(6) * 0.1
    R = np.eye(2) * 0.5

    # Estado inicial (x, y, vx, vy, ax, ay)
    x = np.array([previous_points[0][0], previous_points[0][1], 0, 0, 0, 0], dtype=np.float32)
    P = np.eye(6, dtype=np.float32)

    # F_jacobian
    F_jacobian = np.array([
        [1, 0, dt, 0, 0.5 * dt**2, 0],
        [0, 1, 0, dt, 0, 0.5 * dt**2],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ], dtype=np.float32)

    # Matriz de observação
    H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0]
    ], dtype=np.float32)

    # Alimenta o EKF com os pontos anteriores
    for z in previous_points:
        z = np.array(z, dtype=np.float32)
        # Predição
        x_pred = F_jacobian @ x
        P_pred = F_jacobian @ P @ F_jacobian.T + Q

        # Inovação
        y = z - (H @ x_pred)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Atualização
        x = x_pred + K @ y
        I_KH = np.eye(len(K), dtype=np.float32) - (K @ H)
        P = I_KH @ P_pred

    # Gera previsões futuras
    predictions = []
    for _ in range(prediction_range):
        x_pred = F_jacobian @ x
        predictions.append(x_pred[:2].astype(np.int32))
        x = x_pred

    return predictions


# -----------------------------------------------------------------------------
# 4) CONFIGURAÇÃO DE YOLO E DEEPSORT
# -----------------------------------------------------------------------------
model = YOLO("models/yolo11n.pt").to('cuda')
tracker = DeepSort(
    max_age=1000,
    n_init=5,
    nn_budget=10,
    embedder_gpu=True
)

# -----------------------------------------------------------------------------
# 5) CARREGAR MAPA (lat/lon + HOMOGRAFIA) E TRAJETÓRIAS
# -----------------------------------------------------------------------------
lat0_deg, lon0_deg, homography_mat = load_map_data("map.txt")
predefined_trajectories = load_trajectories("trajetoriasClean.txt")

# -----------------------------------------------------------------------------
# 6) VARIÁVEIS GLOBAIS
# -----------------------------------------------------------------------------
screen_width, screen_height = pyautogui.size()
screen_region = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}

color_map = {}
point_history = {}
missing_track_counter = {}

MIN_DISTANCE_THRESHOLD = 20
COLLISION_THRESHOLD = 30

vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
pedestrian_class = 'person'




# -----------------------------------------------------------------------------
# 7) FUNÇÕES DE APOIO
# -----------------------------------------------------------------------------

def gps2hex(lat: float, lon: float) -> str:
    """
    Converte coordenadas GPS (latitude, longitude) para um inteiro 32-bit representado em hexadecimal.
    O input deve ser um double (float) e o output é o int32 respetivo em hexadecimal.
    
    Utiliza um fator de escala de 1e7 (10.000.000) para preservar 7 casas decimais.
    Retorna uma string formatada como "0xNN 0xNN ..." (4 bytes para cada coordenada).
    """
    scale = 10_000_000  # fator de escala para preservar casas decimais
    # Converter o float para int32 (com arredondamento)
    lat_int = int(round(lat * scale))
    lon_int = int(round(lon * scale))
    
    # Empacotar os inteiros em 4 bytes cada (big-endian)
    lat_bytes = struct.pack('>i', lat_int)
    lon_bytes = struct.pack('>i', lon_int)
    
    # Converter os bytes para uma string hexadecimal
    lat_hex = ' '.join(f'0x{b:02X}' for b in lat_bytes)
    lon_hex = ' '.join(f'0x{b:02X}' for b in lon_bytes)
    
    return f"{lat_hex} {lon_hex}"

def get_random_color():
    """Gera uma cor aleatória em formato BGR."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def is_point_far_enough(new_point, last_point, threshold=MIN_DISTANCE_THRESHOLD):
    """Verifica se a distância entre dois pontos é maior que 'threshold'."""
    return np.linalg.norm(np.array(new_point) - np.array(last_point)) > threshold


# -----------------------------------------------------------------------------
# 8) LOOP PRINCIPAL
# -----------------------------------------------------------------------------
def main():
    with mss() as sct:
        while True:
            start_time = time.perf_counter()
            # 1) Captura de tela
            screenshot = sct.grab(screen_region)
            frame = np.array(screenshot, dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # 2) Detecção com YOLO
            results = model.predict(frame, verbose=False)
            yolo_boxes = results[0].boxes

            # 3) Converter predições YOLO para DeepSort
            detections = []
            for box in yolo_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id))

            # 4) Atualização do tracker
            tracks = tracker.update_tracks(detections, frame=frame)
            current_track_ids = set()

            # Listas para armazenar predições futuras
            future_car_points = []
            future_person_points = []

            # 5) Processar cada track
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                track_id = track.track_id
                current_track_ids.add(track_id)
                missing_track_counter[track_id] = 0

                # Bounding box + centro
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Histórico de pontos
                if track_id not in point_history:
                    point_history[track_id] = deque(maxlen=20)

                # Adiciona ponto se for distante o suficiente
                if not point_history[track_id] or is_point_far_enough(center, point_history[track_id][-1]):
                    point_history[track_id].append(center)

                obj_class = model.names[int(track.get_det_class())]

                # Atribui cor única
                color = color_map.setdefault(track_id, get_random_color())

                # Desenhar bounding box e label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{obj_class} #{track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                # 6) Gera predições futuras
                if obj_class in vehicle_classes:
                    past_points = list(point_history[track_id])[-20:]  # até 20 pts
                    best_traj = find_best_trajectory(past_points, predefined_trajectories, max_points=50)
                    # Desenhar
                    for pt in best_traj:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, color, -1)
                    # Salvar
                    future_car_points.extend(best_traj)

                elif obj_class == pedestrian_class:
                    past_points = list(point_history[track_id])[-10:]
                    pred_points = ekf(past_points, prediction_range=5)
                    # Desenhar
                    for pt in pred_points:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, color, -1)
                    # Salvar
                    future_person_points.extend(pred_points)

            # 7) DETECTAR POSSÍVEIS COLISÕES (EM PIXEL) E CONVERTER P/ GPS
            for car_pt in future_car_points:
                for person_pt in future_person_points:
                    dist = np.linalg.norm(np.array(car_pt) - np.array(person_pt))
                    if dist < COLLISION_THRESHOLD:
                        # (Px, Py) = pixel de colisão
                        Px = int((car_pt[0] + person_pt[0]) / 2)
                        Py = int((car_pt[1] + person_pt[1]) / 2)

                        # 7.1) Desenhar alerta
                        cv2.circle(frame, (Px, Py), 20, (0, 0, 255), -1)

                        # 7.2) Converter pixel -> (X, Y) -> (lat, lon)
                        # [X, Y, W]^T = homography_mat * [Px, Py, 1]^T
                        pt = np.array([[Px], [Py], [1]], dtype=np.float32)
                        XYW = homography_mat @ pt
                        X, Y, W = XYW[0, 0], XYW[1, 0], XYW[2, 0]

                        if abs(W) < 1e-12:
                            print("W muito próximo de zero; resultado instável para conversão.")
                        else:
                            X /= W
                            Y /= W
                            lat_deg, lon_deg = xy_to_latlon(X, Y, lat0_deg, lon0_deg)

                            # 7.3) Imprimir alerta no terminal com LAT/LON
                            print(f"[ALERTA] Possível colisão futura em pixel=({Px},{Py}) "
                                  f"-> lat/lon=({lat_deg:.6f}, {lon_deg:.6f})")
                            
                            # 7.4) Escrever alerta no ficheiro 'data.txt' (escrever pouquinho 5 linhas max)
                            file_path = r"C:\Users\35196\OneDrive\Ambiente de Trabalho\tese\Tese_code\SafeRoadside\shared\data.txt"

                            with open(file_path, "r", encoding="utf-8") as file:
                                lines = file.readlines()

                            if len(lines) < 5:
                                with open(file_path, "a", encoding="utf-8") as file:
                                    file.write(f"{gps2hex(lat_deg, lon_deg)}\n")


            # 8) Exibição
            cv2.imshow("Tracking Inteligente", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            with open("times.txt", "a") as f:
                f.write(f"{processing_time:.3f}\n")

    cv2.destroyAllWindows()


# -----------------------------------------------------------------------------
# PONTO DE ENTRADA
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
