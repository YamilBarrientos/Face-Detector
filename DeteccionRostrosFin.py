from roboflow import Roboflow
import cv2
from collections import Counter

rf = Roboflow(api_key="Id4Ogt9yA8AdOoLgNIJk")
project = rf.workspace().project("drugdispenser")
model = project.version(2).model


# Inicializar el clasificador de rostros de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Lista para almacenar los nombres de clase de cada predicción
class_names = []

# Mostrar la ventana con el frame de la cámara y rectángulo de detección de rostros
while True:
    ret, frame = cap.read()

    # Convertir a escala de grises para la detección de rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Realizar la detección de rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Dibujar un rectángulo alrededor de cada rostro detectado
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Mostrar el frame resultante
    cv2.imshow('Detección de Rostros', frame)

    # Salir del bucle cuando se presiona la tecla 'c'
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

# Reiniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Tomar 5 fotos cuando se detecta un rostro
for i in range(5):
    # Leer un frame desde la cámara
    ret, frame = cap.read()

    # Convertir a escala de grises para la detección de rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Realizar la detección de rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Dibujar un rectángulo alrededor de cada rostro detectado
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Mostrar el frame resultante
    cv2.imshow('Detección de Rostros', frame)

    # Esperar un breve tiempo para que se vea el rectángulo
    cv2.waitKey(500)

    # Tomar la foto
    roi = frame[y:y + h, x:x + w]

    # Obtener la predicción
    prediction = model.predict(f"foto_{i + 1}.jpg", confidence=5, overlap=30).json()
    if prediction.get('predictions'):
        class_name = prediction['predictions'][0]['class']
    else:
        class_name = "Desconocido"

    # Agregar el nombre de la clase a la lista
    class_names.append(class_name)

# Cerrar la ventana después de tomar las 5 fotos
cv2.destroyAllWindows()

# Imprimir las predicciones
print(f"Predicciones para las 5 fotos:")
for i in range(5):
    print(f"Foto {i + 1}: {class_names[i]}")

# Imprimir el nombre de la clase que más se repite
most_common_class = Counter(class_names).most_common(1)[0][0]
print(f"Clase más común: {most_common_class}")

# Liberar la captura de video
cap.release()
