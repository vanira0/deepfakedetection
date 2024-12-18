# data_preparation.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from .video_processor import process_video

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
    os.makedirs(METADATA_DIR, exist_ok=True)

    annotations = []

    for label, video_dir in enumerate([REAL_DIR, FAKE_DIR]):
        if not os.path.exists(video_dir):
            print(f"Warning: Directory {video_dir} does not exist.")
            continue

        for video_file in os.listdir(video_dir):
            video_path = os.path.join(video_dir, video_file)
            try:
                process_video(video_path, label, OUTPUT_DIR, augment=(label == 1))
                output_subdir = "real" if label == 0 else "fake"
                video_output_dir = os.path.join(OUTPUT_DIR, output_subdir, os.path.splitext(video_file)[0])

                if os.path.exists(video_output_dir):
                    for frame_file in os.listdir(video_output_dir):
                        annotations.append([os.path.join(output_subdir, os.path.splitext(video_file)[0], frame_file), label])
            except Exception as e:
                print(f"Error processing video {video_path}: {e}")

    annotations_df = pd.DataFrame(annotations, columns=['frame_path', 'label'])
    annotations_df.to_csv(ANNOTATIONS_FILE, index=False)

    train_df, val_df = train_test_split(annotations_df, test_size=0.2, stratify=annotations_df['label'], random_state=42)

    train_df.to_csv(os.path.join(METADATA_DIR, "train_annotations.csv"), index=False)
    val_df.to_csv(os.path.join(METADATA_DIR, "val_annotations.csv"), index=False)

    print("Dataset creation complete. Annotations and splits saved.")

if __name__ == "__main__":
    prepare_dataset()
