import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import os

# Configuración de la página
st.set_page_config(page_title="Reconocimiento de Lenguaje de Señas", layout="wide")

# Estilo CSS personalizado
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(to right, #f0f2f6, #e0e2e6);
    }
    .big-font {
        font-size: 30px !important;
        font-weight: bold;
        color: #1E3A8A;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #1E3A8A;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    h2 {
        color: #3B82F6;
        font-family: 'Arial', sans-serif;
    }
    .box {
        border: 4px solid #2563EB; /* Azul bonito */
        padding: 10px;
        margin-top: 10px;
        background-color: #e0f2ff;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        min-height: 50px; /* Para asegurar espacio suficiente para el texto */
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .recognized-text {
        font-size: 28px;
        color: #1E3A8A;
        font-weight: bold;
        font-family: 'Courier New', Courier, monospace;
        text-align: center;
        margin: 0; /* Quitar margen alrededor del texto */
    }
    .clean-button {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Título principal con icono
st.markdown("<h1>🤟 Reconocimiento de Lenguaje de Señas</h1>", unsafe_allow_html=True)

# Cargar el modelo y los datos
@st.cache_resource
def load_model_and_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dict = pickle.load(open(os.path.join(script_dir, 'model.p'), 'rb'))
    with open(os.path.join(script_dir, 'data.pickle'), 'rb') as f:
        data_dict = pickle.load(f)
    return model_dict['model'], data_dict

model, data_dict = load_model_and_data()

# Configuración de MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Diccionario de etiquetas
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: ' '
}

# Variables globales
current_word = ""
last_prediction = None
prediction_start_time = time.time()
PREDICTION_THRESHOLD = 1  # 1 segundo

# Función principal de procesamiento
def process_frame(frame):
    global current_word, last_prediction, prediction_start_time

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

        data_aux = []
        x_ = []
        y_ = []

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

        x1 = int(min(x_) * frame.shape[1]) - 10
        y1 = int(min(y_) * frame.shape[0]) - 10
        x2 = int(max(x_) * frame.shape[1]) - 10
        y2 = int(max(y_) * frame.shape[0]) - 10

        data_aux = data_aux[:84]  # Truncar si hay más de 84
        data_aux = data_aux + [0] * (84 - len(data_aux))  # Rellenar con ceros si hay menos de 84

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        if predicted_character != last_prediction:
            prediction_start_time = time.time()
            last_prediction = predicted_character
        elif time.time() - prediction_start_time >= PREDICTION_THRESHOLD:
            if not current_word or predicted_character != current_word[-1]:
                current_word += predicted_character
                prediction_start_time = time.time()

    else:
        last_prediction = None
        prediction_start_time = time.time()

    return frame

# Interfaz de Streamlit
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<h2>Texto Reconocido</h2>", unsafe_allow_html=True)
    with st.container():
        text_display = st.markdown('<div class="box"><div class="recognized-text"></div></div>', unsafe_allow_html=True)

with col2:
    st.markdown("<h2>Reconocimiento en Vivo</h2>", unsafe_allow_html=True)
    live_feed = st.empty()

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Bucle principal
def main():
    global current_word
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("No se pudo acceder a la cámara.")
            break

        frame = cv2.resize(frame, (480, 360))
        processed_frame = process_frame(frame)

        live_feed.image(processed_frame, channels="BGR", use_column_width=True)
        text_display.markdown(f'<div class="box"><div class="recognized-text">{current_word}</div></div>', unsafe_allow_html=True)

        # Nota: No podemos usar cv2.waitKey en Streamlit, así que omitimos esa parte

if st.button("Iniciar Reconocimiento"):
    main()

# Liberar recursos
cap.release()

# Botón de limpiar texto abajo
st.markdown('<div class="clean-button">', unsafe_allow_html=True)
clear_button = st.button("🧹 Limpiar Texto")
if clear_button:
    current_word = ""
st.markdown('</div>', unsafe_allow_html=True)

# Añadir una sección de información
st.markdown("---")
st.markdown("""
<h3 style='text-align: center; color: #4B5563;'>Sobre esta aplicación</h3>
<p style='text-align: center; color: #4B5563;'>
Esta aplicación utiliza inteligencia artificial para reconocer el lenguaje de señas en tiempo real.
Simplemente realiza los gestos frente a la cámara y observa cómo se traduce a texto.
</p>
""", unsafe_allow_html=True)
