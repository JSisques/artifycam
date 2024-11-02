import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo
model = load_model('emotion_classifier_model.h5')

# Diccionario de etiquetas
emotion_labels = {
    0: "Enojo",
    1: "Desprecio",
    2: "Feliz",
    3: "Triste",
    4: "Sorpresa",
    5: "Desprecio",
    6: "Neutral"
}

# Inicializar la captura de video
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extraer la región de interés
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float32') / 255.0
        roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))

        # Hacer la predicción
        predictions = model.predict(roi_gray)
        emotion_index = np.argmax(predictions[0])
        emotion_label = emotion_labels[emotion_index]

        # Dibujar un rectángulo y mostrar la etiqueta
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Mostrar el video con las emociones
    cv2.imshow('Clasificador de Emociones', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video
video_capture.release()
cv2.destroyAllWindows()
