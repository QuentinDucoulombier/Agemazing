import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from imutils.video import VideoStream
import imutils
import dlib

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# Function to load the model and set dependencies
def load_selected_model(model_choice):
    if model_choice == 1:
        model_path = 'age_resnet_final(1).h5'
        from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore
        image_size = [224, 224]
    elif model_choice == 2:
        model_path = 'age_xception_final.h5'
        from tensorflow.keras.applications.xception import preprocess_input # type: ignore
        image_size = [299, 299]
    elif model_choice == 3:
        model_path = 'age_inceptionresnetv2_overfitting.h5'
        from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input # type: ignore
        image_size = [299, 299]
    elif model_choice == 4:
        model_path = 'age_mobileNetV3_nooverfitting.h5'
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input # type: ignore
        image_size = [224, 224]
    else:
        raise ValueError("Invalid model choice. Please choose 1, 2, 3 or 4.")
    
    model = load_model(model_path)
    return model, preprocess_input, image_size

# Select the model
print("Choose a model to use:")
print("1. ResNet (file age_Resnet.h5)")
print("2. Xception (file age_Xception.h5)")
print("3. Inception ResNet v2 (file age_InceptionResnet.h5)")
print("4. MobileNetV3 (file age_MobileNetV3.h5)")
model_choice = int(input("Enter the number of the model you want to use: "))

model, preprocess_input, IMAGE_SIZE = load_selected_model(model_choice)

# List of class labels
labels = ['baby', 'child', 'student', 'young adult', 'adult', 'senior']

# Load OpenCV's deep learning face detector
prototxt_path = 'deploy.prototxt'
model_path = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Load dlib's face detector and shape predictor for alignment
predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def align_face(image, left_eye, right_eye, nose):
    left_eye_center = np.array(left_eye, dtype=np.float32)
    right_eye_center = np.array(right_eye, dtype=np.float32)
    
    # Calculate the center point between the two eyes
    eyes_center = ((left_eye_center[0] + right_eye_center[0]) * 0.5, 
                   (left_eye_center[1] + right_eye_center[1]) * 0.5)
    
    # Calculate the angle between the eye line and the horizontal
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx)) - 180

    # Ensure the angle is between -90 and 90
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    
    # Apply the affine transformation
    output = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
    
    return output


def preprocess_frame(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

# Set the window to full screen
cv2.namedWindow('Age Prediction', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Age Prediction', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    frame = cap.read()
    if frame is None:
        break

    # Resize the frame for faster processing
    frame = imutils.resize(frame, width=800)

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

         # Convert face to grayscale
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Detect landmarks
        rect = dlib.rectangle(0, 0, face.shape[1], face.shape[0])
        shape = predictor(gray, rect)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        # Get coordinates of the eyes and nose
        left_eye = np.mean(shape[36:42], axis=0).astype("int")
        right_eye = np.mean(shape[42:48], axis=0).astype("int")
        nose = np.mean(shape[27:35], axis=0).astype("int")

        # # Align face
        aligned_face = align_face(face, left_eye, right_eye, nose)


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

        # Save the aligned face image
        cv2.imwrite("detected_face_aligned.jpg", aligned_face)
        
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
cap.stream = None
