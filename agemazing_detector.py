import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import time
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# Fonction pour charger le modèle et définir les dépendances
def load_selected_model(model_choice):
    if model_choice == 1:
        model_path = 'age_Resnet.h5'
        from tensorflow.keras.applications.resnet50 import preprocess_input
        image_size = [100, 100]
        #tester a un moment avec 224, 224
        #image_size = [224, 224]
    elif model_choice == 2:
        model_path = 'age_Xception.h5'
        from tensorflow.keras.applications.xception import preprocess_input
        image_size = [299, 299]
    elif model_choice == 3:
        model_path = 'age_InceptionResnet.h5'
        from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
        image_size = [299, 299]
    else:
        raise ValueError("Invalid model choice. Please choose 1, 2, or 3.")
    
    model = load_model(model_path)
    return model, preprocess_input, image_size

# Sélection du modèle
print("Choose a model to use:")
print("1. ResNet (fichier age_Resnet.h5)")
print("2. Xception (fichier age_Xception.h5)")
print("3. Inception ResNet v2 (fichier age_InceptionResnet.h5)")
model_choice = int(input("Enter the number of the model you want to use: "))

model, preprocess_input, IMAGE_SIZE = load_selected_model(model_choice)

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
    
    # Appliquer le prétraitement spécifique au modèle choisi
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
    time.sleep(0.5)

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
