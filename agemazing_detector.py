import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt
import time

# Charger le modèle pré-entrainé
model = load_model('age.h5')  # Remplacer par le chemin de votre modèle

# Taille de l'image
IMAGE_SIZE = [100, 100]

# Liste des étiquettes de classes
labels = ['001-004', '005-011', '012-018', '019-034', '035-044', '045-064', '065-110']

# Charger le classificateur de visage de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_frame(frame):
    # Convertir l'image en RGB si ce n'est pas déjà fait
    if frame.shape[2] == 3:  # Si l'image a 3 canaux
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Redimensionner l'image
    frame = cv2.resize(frame, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
    
    # Appliquer le prétraitement spécifique à ResNet50
    frame = preprocess_input(frame)
    
    # Ajouter une dimension pour correspondre au batch_size
    frame = frame.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    
    return frame

def predict_age(face):
    # Prétraiter l'image
    processed_face = preprocess_frame(face)
    
    # Faire la prédiction
    prediction = model.predict(processed_face)
    
    # Obtenir l'âge prédit
    predicted_age = labels[np.argmax(prediction)]
    pred_list = {x: float(y) for x, y in zip(labels, prediction[0])}
    pred_list = dict(sorted(pred_list.items(), reverse=True, key=lambda item: item[1]))
    print(pred_list)
    
    return predicted_age

# Capture vidéo depuis la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image de BGR à RGB pour la détection de visage
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Détecter les visages
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extraire le visage de l'image
        face = frame[y:y+h, x:x+w]
        
        # Prédire l'âge
        age = predict_age(face)
        
        # Afficher la prédiction sur l'image
        cv2.putText(frame, f'Predicted Age: {age}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Dessiner un rectangle autour du visage détecté
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Afficher l'image
    cv2.imshow('Age Prediction', frame)

    # Sortir de la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Ajouter un délai pour réduire le FPS
    #time.sleep(0.5)

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
