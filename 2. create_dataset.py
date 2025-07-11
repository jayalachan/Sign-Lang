import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue
    for img_path in os.listdir(dir_path):
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(dir_path, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

# Convierte las etiquetas a enteros
label_map = {label: int(label) for label in set(labels)}
labels = [label_map[label] for label in labels]

# Guarda los datos y las etiquetas
output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.pickle')
with open(output_file, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels, 'label_map': label_map}, f)

print(f"Datos guardados en {output_file}")
print(f"Número de muestras guardadas: {len(data)}")
print(f"Etiquetas únicas guardadas: {set(labels)}")