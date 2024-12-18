import cv2
import os
from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, GaussianBlur
from .face_detector import detect_faces, extract_face

def get_augmentation_pipeline():
    return Compose([
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.3),
        GaussianBlur(p=0.3)
    ])

def process_video(video_path, label, output_dir, augment=False):
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    aug_pipeline = get_augmentation_pipeline() if augment else None

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.join(output_dir, "real" if label == 0 else "fake", video_name)
    os.makedirs(video_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray)
            for i, face_rect in enumerate(faces):
                face = extract_face(frame, face_rect)
                
                if face.size > 0:
                    face = cv2.resize(face, (224, 224))

                    if aug_pipeline:
                        face = aug_pipeline(image=face)['image']

                    frame_filename = f"{frame_count}_{i}.jpg"
                    frame_path = os.path.join(video_dir, frame_filename)
                    cv2.imwrite(frame_path, face)

        frame_count += 1

    cap.release()

# Remove the detect_faces function from this file
