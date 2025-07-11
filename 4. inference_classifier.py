import pickle
import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
import time

# Cambia al directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("Directorio actual:", os.getcwd())
print("Contenido del directorio:", os.listdir())

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Carga también el data_dict para obtener la longitud máxima
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

max_length = max(len(d) for d in data_dict['data'])
print("Longitud máxima de los elementos de datos:", max_length)

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: ' '
}

# Initialize pygame for the text window
pygame.init()
text_window = pygame.display.set_mode((400, 200))
pygame.display.set_caption("Signed Words")
font = pygame.font.Font(None, 36)

current_word = ""
last_prediction = None
prediction_start_time = time.time()  # Inicializa con el tiempo actual
PREDICTION_THRESHOLD = 1  # 1 segundos

running = True
while running:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

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

        # Rellena data_aux con ceros si es necesario
        data_aux = data_aux + [0] * (max_length - len(data_aux))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        try:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
        except Exception as e:
            print(f"Error en la predicción: {e}")
            predicted_character = "?"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

        # Lógica para mantener la predicción por 3 segundos
        if predicted_character != last_prediction:
            prediction_start_time = time.time()
            last_prediction = predicted_character
        elif time.time() - prediction_start_time >= PREDICTION_THRESHOLD:
            if not current_word or predicted_character != current_word[-1]:
                current_word += predicted_character
                prediction_start_time = time.time()  # Reinicia el tiempo para la siguiente letra

    else:
        last_prediction = None
        prediction_start_time = time.time()  # Reinicia el tiempo cuando no se detecta mano

    # Mostrar información de depuración
    cv2.putText(frame, f"Longitud de datos: {len(data_aux)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    # Update the text window
    text_window.fill((255, 255, 255))  # White background
    text_surface = font.render(current_word, True, (0, 0, 0))  # Black text
    text_window.blit(text_surface, (10, 10))
    pygame.display.flip()

    # Manejar eventos de Pygame
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                current_word = ""  # Reiniciar el texto al presionar 'Q'
            elif event.key == pygame.K_a:
                running = False  # Terminar el programa al presionar 'A'

    key = cv2.waitKey(1)
    if key & 0xFF == ord('a'):
        running = False

cap.release()
cv2.destroyAllWindows()
pygame.quit()