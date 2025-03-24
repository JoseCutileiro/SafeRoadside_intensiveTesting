import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pyautogui
import os
import random

# Load YOLO model and tracker
model = YOLO("models/yolo11n.pt").to('cuda')
tracker = DeepSort(max_age=1000, n_init=1, nn_budget=50, embedder_gpu=True)

# Screen capture settings
screen_width, screen_height = pyautogui.size()
screen_region = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}

# Dictionary to store tracked points
point_history = {}
track_colors = {}
vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']

def get_track_color(track_id):
    """Ensure each track_id has a unique color."""
    if track_id not in track_colors:
        track_colors[track_id] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    return track_colors[track_id]

# Load road image
if not os.path.exists("road.png"):
    print("road.png not found!")
    exit()
image = cv2.imread("road.png")

# Live tracking loop
with mss() as sct:
    try:
        while True:
            screenshot = sct.grab(screen_region)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            results = model.predict(frame)
            detections = []
            for r in results[0].boxes:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                confidence = float(r.conf[0])
                class_id = int(r.cls[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id))
            
            tracks = tracker.update_tracks(detections, frame=frame)
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                obj = model.names[int(track.get_det_class())]
                if obj in vehicle_classes:
                    if track_id not in point_history:
                        point_history[track_id] = []
                    point_history[track_id].append(center)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{obj} #{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            cv2.imshow("AutoDraw - Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

# Draw detected points on road image
def draw_trajectories():
    """Overlay detected points on road.png and save as withguide.png."""
    for track_id, points in point_history.items():
        color = get_track_color(track_id)
        for point in points:
            cv2.circle(image, point, 3, color, -1)  # Draw small colored points
    
    cv2.imwrite("withguide.png", image)
    print("Detected points drawn and saved as withguide.png")

draw_trajectories()