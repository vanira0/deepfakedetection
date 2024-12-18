import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf

METADATA_DIR = "./data/metadata"
OUTPUT_DIR = "./data/processed"

class DeepfakeDataset(tf.keras.utils.Sequence):
    def __init__(self, annotation_file, batch_size, image_size=(224, 224), max_frames=30, shuffle=True):
        self.annotation_file = annotation_file
        self.batch_size = batch_size
        self.image_size = image_size
        self.max_frames = max_frames
        self.shuffle = shuffle

        # Load annotations
        self.annotations = pd.read_csv(annotation_file)
        self.indexes = np.arange(len(self.annotations))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.annotations) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.annotations.iloc[batch_indexes]
        X, y = self.__data_generation(batch_data)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_data):
        X = np.zeros((len(batch_data), self.max_frames, *self.image_size, 3), dtype=np.float32)
        y = np.zeros(len(batch_data), dtype=np.float32)

        for i, (_, row) in enumerate(batch_data.iterrows()):
            normalized_path = row['frame_path'].replace('\\', '/')
            video_frames_directory = os.path.dirname(normalized_path)
            video_path = os.path.join(OUTPUT_DIR, video_frames_directory)
            # video_path = os.path.join(OUTPUT_DIR, normalized_path)
            # video_path = os.path.normpath(video_path) 
            label = row['label']

            frames = self.load_frames(video_path)
            X[i, :len(frames)] = frames
            y[i] = label

        return X, tf.keras.utils.to_categorical(y, num_classes=2)

    def load_frames(self, folder_path):
        frames = []
        image_files = sorted(os.listdir(folder_path))
        frame_count = 0

        for image_file in image_files:
            if frame_count >= self.max_frames:
                break

            image_path = os.path.join(folder_path, image_file)
            frame = cv2.imread(image_path)

            if frame is None:
                continue

            frame = cv2.resize(frame, self.image_size)
            frame = frame / 255.0
            frames.append(frame)
            frame_count += 1

        if len(frames) < self.max_frames:
            padding = [np.zeros((self.image_size[0], self.image_size[1], 3))] * (self.max_frames - len(frames))
            frames.extend(padding)

        return np.array(frames)


def create_train_val_loaders(train_csv, val_csv, batch_size=8, image_size=(224, 224), max_frames=10):
    train_loader = DeepfakeDataset(train_csv, batch_size, image_size, max_frames, shuffle=True)
    val_loader = DeepfakeDataset(val_csv, batch_size, image_size, max_frames, shuffle=False)

    return train_loader, val_loader
