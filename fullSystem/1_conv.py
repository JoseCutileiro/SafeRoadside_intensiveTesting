#!/usr/bin/env python3
# conv.py

import cv2
import numpy as np
import math

clicked_point = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Pixel clicado = ({x}, {y})")

def xy_to_latlon(X, Y, lat0_deg, lon0_deg):
    """
    Inverso da projeção local.
    Dado (X, Y) em metros no sistema local,
    devolve (lat, lon) em graus decimais.
    """
    R = 6371000.0
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)

    lat = Y / R + lat0
    lon = X / (R * math.cos(lat0)) + lon0

    # Convert para graus
    lat_deg = math.degrees(lat)
    lon_deg = math.degrees(lon)
    return lat_deg, lon_deg

def main():
    global clicked_point

    # Carregar a imagem
    image_path = r"tes.png"
    img = cv2.imread(image_path)
    if img is None:
        print("Erro ao carregar imagem.")
        return

    # Ler a info de map.txt
    try:
        with open("map.txt", "r") as f:
            lines = f.readlines()
        # Formato esperado:
        # linha 1: lat0_deg lon0_deg
        # linha 2..4: h11 h12 h13
        #            h21 h22 h23
        #            h31 h32 h33
        lat0_deg, lon0_deg = map(float, lines[0].split())

        # Lê as próximas 3 linhas e converte para float
        h_rows = []
        for i in range(1, 4):
            row_vals = list(map(float, lines[i].split()))
            h_rows.append(row_vals)

        homography_mat = np.array(h_rows, dtype=np.float32)  # shape (3, 3)
    except Exception as e:
        print("Erro a ler map.txt:", e)
        return

    if homography_mat.shape != (3, 3):
        print("Matriz de homografia não tem dimensões 3x3.")
        return

    # Preparar janela
    cv2.namedWindow("Clica no ponto para ver lat/lon")
    cv2.setMouseCallback("Clica no ponto para ver lat/lon", mouse_callback)

    print("Instruções: clica em qualquer ponto para converter pixel -> lat/lon.\n"
          "Pressiona ESC para sair.")

    while True:
        disp = img.copy()
        cv2.imshow("Clica no ponto para ver lat/lon", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

        if clicked_point is not None:
            px, py = clicked_point

            # Aplicar a homografia para obter (X, Y) local
            # [X, Y, W]^T = H * [px, py, 1]^T
            pt = np.array([[px], [py], [1]], dtype=np.float32)
            XYW = homography_mat @ pt
            X = XYW[0, 0]
            Y = XYW[1, 0]
            W = XYW[2, 0]

            if abs(W) < 1e-12:
                print("W muito próximo de zero; resultado instável.")
            else:
                X /= W
                Y /= W

            # Converter (X, Y) local -> (lat, lon)
            lat_deg, lon_deg = xy_to_latlon(X, Y, lat0_deg, lon0_deg)

            print(f"\nPixel=({px}, {py})"
                  f"\n -> local (X, Y) = ({X:.3f}, {Y:.3f})"
                  f"\n -> lat/lon = ({lat_deg:.6f}, {lon_deg:.6f})\n")

            # Reset para permitir novo clique
            clicked_point = None

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
