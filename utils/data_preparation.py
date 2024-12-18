import os
import pandas as pd
from sklearn.model_selection import train_test_split
# from .video_processor import process_video
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from utils.video_processor import process_video
from utils.video_processor import process_video


DATASET_DIR = "./data/raw/"
OUTPUT_DIR = "./data/processed/"
METADATA_DIR = "./data/metadata/"

REAL_DIR = os.path.join(DATASET_DIR, "real")
FAKE_DIR = os.path.join(DATASET_DIR, "fake")
ANNOTATIONS_FILE = os.path.join(METADATA_DIR, "annotations.csv")

def prepare_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "real"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "fake"), exist_ok=True)

    annotations = []

    for label, video_dir in enumerate([REAL_DIR, FAKE_DIR]):
        for video_file in os.listdir(video_dir):
            video_path = os.path.join(video_dir, video_file)
            process_video(video_path, label, OUTPUT_DIR, augment=(label == 1))

            for frame_file in os.listdir(os.path.join(OUTPUT_DIR, "real" if label == 0 else "fake")):
                annotations.append([os.path.join("real" if label == 0 else "fake", frame_file), label])

    annotations_df = pd.DataFrame(annotations, columns=['video_path', 'label'])
    annotations_df.to_csv(ANNOTATIONS_FILE, index=False)

    train_df, val_df = train_test_split(annotations_df, test_size=0.2, stratify=annotations_df['label'], random_state=42)

    train_df.to_csv(os.path.join(METADATA_DIR, "train_annotations.csv"), index=False)
    val_df.to_csv(os.path.join(METADATA_DIR, "val_annotations.csv"), index=False)

    print("Dataset creation complete. Annotations and splits saved.")

if __name__ == "__main__":
    prepare_dataset()
