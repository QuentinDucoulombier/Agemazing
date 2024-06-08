import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from imutils.video import VideoStream
import imutils

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# Function to load the model and set dependencies
def load_selected_model(model_choice):
    if model_choice == 1:
        model_path = 'age_Resnet.h5'
        from tensorflow.keras.applications.resnet50 import preprocess_input
        image_size = [100, 100]
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

# Select the model
print("Choose a model to use:")
print("1. ResNet (file age_Resnet.h5)")
print("2. Xception (file age_Xception.h5)")
print("3. Inception ResNet v2 (file age_InceptionResnet.h5)")
model_choice = int(input("Enter the number of the model you want to use: "))

model, preprocess_input, IMAGE_SIZE = load_selected_model(model_choice)

# List of class labels
labels = ['001-004', '005-011', '012-018', '019-034', '035-044', '045-064', '065-110']

# Load OpenCV's deep learning face detector
prototxt_path = 'deploy.prototxt'
model_path = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# def apply_clahe(image):
#     # Convert the image to LAB color space
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)

#     # Apply CLAHE to the L channel
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     cl = clahe.apply(l)

#     # Merge the CLAHE enhanced L channel with the a and b channels
#     limg = cv2.merge((cl, a, b))

#     # Convert the image back to BGR color space
#     final_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
#     return final_image

def preprocess_frame(frame):
    # Resize the image
    frame = cv2.resize(frame, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
    
    # Apply the preprocessing specific to the chosen model
    frame = preprocess_input(frame)
    
    # Add a dimension to match the batch size
    frame = frame.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    
    return frame

def predict_age(face):
    # Preprocess the image
    processed_face = preprocess_frame(face)
    
    # Make the prediction
    prediction = model.predict(processed_face)
    
    # Get the predicted age
    predicted_age = labels[np.argmax(prediction)]
    pred_list = {x: float(y) for x, y in zip(labels, prediction[0])}
    pred_list = dict(sorted(pred_list.items(), reverse=True, key=lambda item: item[1]))
    print(pred_list)
    
    return predicted_age

def get_square_box(startX, startY, endX, endY, frame_shape):
    # Calculate width and height of the bounding box
    width = endX - startX
    height = endY - startY

    # Determine the size of the square box
    max_dim = max(width, height)

    # Calculate new start and end points for the square box
    centerX = startX + width // 2
    centerY = startY + height // 2

    new_startX = max(centerX - max_dim // 2, 0)
    new_startY = max(centerY - max_dim // 2, 0)
    new_endX = min(centerX + max_dim // 2, frame_shape[1])
    new_endY = min(centerY + max_dim // 2, frame_shape[0])

    return new_startX, new_startY, new_endX, new_endY

# Capture video from the webcam
cap = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = cap.read()
    if frame is None:
        break

    # Resize the frame for faster processing
    frame = imutils.resize(frame, width=400)

    # Convert the frame dimensions
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and get the detections
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (probability)
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence < 0.5:
            continue

        # Compute the (x, y)-coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Adjust the bounding box to be a square
        startX, startY, endX, endY = get_square_box(startX, startY, endX, endY, frame.shape)

        # Extract the face ROI
        face = frame[startY:endY, startX:endX]

        # Predict the age
        age = predict_age(face)

        # Display the prediction on the image
        text = f'Predicted Age: {age}'
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)

    # Show the output frame
    cv2.imshow('Age Prediction', frame)

    # Capture a photo of the detected face if 'A' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        # Save the original face image
        cv2.imwrite("detected_face_original.jpg", face)
        
        # Save the CLAHE-processed face image
        # face_clahe = apply_clahe(face)
        # cv2.imwrite("detected_face_clahe.jpg", face_clahe)

    # Break from the loop if the 'q' key was pressed
    elif key == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
cap.stop()
cap.stream.release()
