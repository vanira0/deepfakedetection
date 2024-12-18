# face_detector.py

import cv2
import numpy as np

def detect_faces(image):
    # Check if the image is already grayscale
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def extract_face(image, face_rect):
    x, y, w, h = face_rect
    return image[y:y+h, x:x+w]
