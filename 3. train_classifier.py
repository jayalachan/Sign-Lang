import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os

# Cambia al directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("Directorio actual:", os.getcwd())
print("Contenido del directorio:", os.listdir())

# Carga los datos
data_file = 'data.pickle'
print(f"Intentando cargar datos desde: {os.path.abspath(data_file)}")

with open(data_file, 'rb') as f:
    data_dict = pickle.load(f)

print("Claves en data_dict:", data_dict.keys())

# Analiza la estructura de los datos
print("Estructura de los datos:")
print("Tipo de data:", type(data_dict['data']))
print("Longitud de data:", len(data_dict['data']))
print("Tipo del primer elemento:", type(data_dict['data'][0]))
print("Longitud del primer elemento:", len(data_dict['data'][0]))

# Encuentra la longitud máxima de los elementos de datos
max_length = max(len(d) for d in data_dict['data'])
print("Longitud máxima de los elementos de datos:", max_length)

# Rellena los elementos más cortos con ceros
data = np.array([d + [0] * (max_length - len(d)) for d in data_dict['data']])
labels = np.array(data_dict['labels'])

print("Datos procesados:")
print("Forma de data:", data.shape)
print("Forma de labels:", labels.shape)
print("Etiquetas únicas:", set(labels))
print("Número de etiquetas únicas:", len(set(labels)))

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

# Specify the full path for the output file
output_file = os.path.join(script_dir, 'model.p')

with open(output_file, 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model has been saved to 'model.p'.")