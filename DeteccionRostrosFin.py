from roboflow import Roboflow
import cv2
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

rf = Roboflow(api_key="Id4Ogt9yA8AdOoLgNIJk")
project = rf.workspace().project("drugdispenser")
model = project.version(2).model


# Inicializar el clasificador de rostros de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Lista para almacenar los nombres de clase de cada predicción
class_names = []
#función para el dataset
def consultar_paciente(nombre_paciente, archivo_excel='dataset_pacientes.xlsx'):
    # Cargar el archivo Excel
    df = pd.read_excel(archivo_excel)

    # Buscar el paciente en el DataFrame
    fila_paciente = df[df['Nombre'] == nombre_paciente]

    # Verificar si el paciente está en la lista
    if not fila_paciente.empty:
        medicamento = fila_paciente['Medicamento'].values[0]
        cantidad = fila_paciente['Cantidad por Día'].values[0]
        mensaje = f"{nombre_paciente} debe tomar {cantidad} {medicamento} por día."
        return mensaje, medicamento
    else:
        return "PACIENTE NO REGISTRADO", None


#Clase para la dispensación:

def activar_motor_segun_medicamento(nombre_medicamento):
    
    if nombre_medicamento.lower() == 'ibuprofeno':
        # Código para activar el motor 1
        mensaje = "Dispensando Ibuprofeno"
    elif nombre_medicamento.lower() == 'paracetamol':
        # Código para activar el motor 2
        mensaje = "Dispensando Paracetamol"
    elif nombre_medicamento.lower() == 'amoxicilina':
        # Código para activar el motor 3
        mensaje = "Dispensando Amoxicilina"
    else:
        mensaje = None
    return mensaje
        
#funciones para el stock
def graficar(base_image,thresh,opening,sure_bg,dist_transform,sure_fg,unknown,markers,watershed):
  # Show the pictures
  plt.rcParams["figure.figsize"] = (20,15)
  # First row
  fig, ax = plt.subplots(3, 3)
  ax[0][0].imshow(base_image)
  ax[0][0].set_title("Original")
  ax[0][1].imshow(thresh, cmap = "gray")
  ax[0][1].set_title("Otsu")
  ax[0][2].imshow(opening, cmap = "gray")
  ax[0][2].set_title("2 Iterations of Closing + 1 Iteration of Opening")

  # Second row
  ax[1][0].imshow(sure_bg, cmap = "gray")
  ax[1][0].set_title("Sure Background (Black region)")
  ax[1][1].imshow(dist_transform, cmap = "gray")
  ax[1][1].set_title("Distance Transform")
  ax[1][2].imshow(sure_fg, cmap = "gray")
  ax[1][2].set_title("Sure Foreground")

  # Third row
  ax[2][0].imshow(unknown, cmap = "gray")
  ax[2][0].set_title("Unkown ")
  ax[2][1].imshow(markers, cmap = "jet")
  ax[2][1].set_title("Markers")
  ax[2][2].imshow(base_image)
  ax[2][2].imshow(watershed, cmap = "winter",alpha=0.3)
  ax[2][2].set_title("Overlapped")
  
def stock_pastillas(path):
  image = cv2.imread(path)
  image = cv2.resize(image, (550,550))
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  base_image = np.copy(image)
  grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Aplicamos Otsu
  ret, thresh = cv2.threshold(grayscale,0,255,cv2.THRESH_OTSU)
  # usamos closing y opening para eliminar ruido
  structuring_element = np.ones((3,3),np.uint8)
  closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, structuring_element, iterations = 2)
  opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, structuring_element, iterations = 1)

  # Find the area that belong to the background
  sure_bg = cv2.dilate(opening, structuring_element, iterations=1)

  # Find the area of the foreground
  # DistanceTransform replaces the value of each pixel with its distance to the nearest background pixel
  # It Receives as parameters: a binary image, function of distance, kernel size
  dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
  ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,cv2.THRESH_BINARY)
  sure_fg = np.uint8(sure_fg)

  # Find the unknown region
  unknown = cv2.subtract(sure_bg,sure_fg)

  # Label initial regions (markers)
  ret, markers = cv2.connectedComponents(sure_fg)
  # Add one to all tags so the safe background isn't confused with the tag marker 1
  markers = markers+1
  # Define the unknown region with zero pixels
  markers[unknown==255] = 0


  # Apply Watershed Segmentation
  watershed = cv2.watershed(image,markers)
  # Watershed assigns the value -1 to region-bounding pixels, so
  # we will assign the color red
  image[watershed == -1] = [255,0,0]

  #graficar(base_image,thresh,opening,sure_bg,dist_transform,sure_fg,unknown,markers,watershed)

  return watershed.max()
        
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
    prediction = model.predict(f"foto_{i + 1}.jpg", confidence=20, overlap=30).json()
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
if most_common_class == 'Desconocido':
    print(f"Paciente DESCONOCIDO, no se puede dispensar medicamento")
else:
    #Consultar si el paciente pertecene o no al sistema
    mensaje, nombre_medicamento = consultar_paciente(most_common_class)
    print(mensaje)
    stock = stock_pastillas("pastillas.png")
    #stock = 0
    print("El numero de pastillas disponibles es de es de: "+str(stock))
    if  stock == 0:
        print("LO SIENTO SE NOS ACABARON LAS PASTILLAS QUE NECESITAS")
    else:
        #Llamar a la función de dispensacion:
        mensaje_motor = activar_motor_segun_medicamento(nombre_medicamento)
        print(mensaje_motor)

# Liberar la captura de video
cap.release()





