import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the image
img_path = 'path_to_your_image.jpg'  # Replace with your image path
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array_expanded = np.expand_dims(img_array, axis=0)
img_preprocessed = preprocess_input(img_array_expanded)

# Load MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Predict using MobileNetV2
predictions = model.predict(img_preprocessed)
decoded_preds = decode_predictions(predictions, top=3)[0]

print("🔍 Image Recognition (MobileNetV2):")
for i, (imagenet_id, label, confidence) in enumerate(decoded_preds):
    print(f"{i + 1}. {label}: {confidence:.2f}")

# Load original image with OpenCV for face detection
cv_img = cv2.imread(img_path)
gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

# Face detection using Haar cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

print(f"\n😀 Faces detected: {len(faces)}")

# Draw bounding boxes around faces
for (x, y, w, h) in faces:
    cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Convert BGR to RGB for matplotlib
cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

# Show the image with detected faces
plt.imshow(cv_img_rgb)
plt.title("Face Detection")
plt.axis('off')
plt.show()
