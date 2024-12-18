import pandas as pd
import numpy as np
import cv2
import os
# import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

METADATA_DIR = "../data/metadata"
OUTPUT_DIR = "../data/processed"

class DeepfakeDataset:
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
            video_path = os.path.join(OUTPUT_DIR, row['video_path']) # output directory??
            label = row['label']

            frames = self.load_frames(video_path)
            X[i, :len(frames)] = frames
            y[i] = label

        return X, y

    def load_frames(self, folder_path):
        # print(f"Loading frames from folder: {folder_path}")
        frames = []
        image_files = sorted(os.listdir(folder_path))
        frame_count = 0

        for image_file in image_files:
            if frame_count >= self.max_frames:
                break
            
            image_path = os.path.join(folder_path, image_file)
            frame = cv2.imread(image_path)

            if frame is None:
                # print(f"Failed to load image: {image_path}")
                continue

            frame = cv2.resize(frame, self.image_size)
            frame = frame / 255.0
            frames.append(frame)
            frame_count += 1

        if len(frames) < self.max_frames:
            padding = [np.zeros((self.image_size[0], self.image_size[1], 3))] * (self.max_frames - len(frames))
            frames.extend(padding)

        # print(f"Successfully loaded {len(frames)} frames from folder: {folder_path}")
        return np.array(frames)


def load_data(annotation_file, batch_size=8, image_size=(224, 224), max_frames=30):
    dataset = DeepfakeDataset(annotation_file, batch_size, image_size, max_frames)
    return dataset


def create_train_val_loaders(train_csv, val_csv, batch_size=8, image_size=(224, 224), max_frames=10):
    train_loader = DeepfakeDataset(train_csv, batch_size, image_size, max_frames, shuffle=True)
    val_loader = DeepfakeDataset(val_csv, batch_size, image_size, max_frames, shuffle=False)
    
    return train_loader, val_loader


def preprocess_data(X, y):
    # Convert labels to categorical
    y = to_categorical(y)
    
    # Normalize pixel values
    X = X.astype('float32') / 255.0
    
    return X, y


def get_class_weights(y):
    class_counts = np.sum(y, axis=0)
    total_samples = sum(class_counts)
    class_weights = {}
    for i, count in enumerate(class_counts):
        class_weights[i] = total_samples / (count * len(np.unique(y)))
    return class_weights


# # Example usage
# if __name__ == "__main__":
#     OUTPUT_DIR = 'path/to/your/output/directory'
    
#     # Load data
#     train_dataset = load_data('train_annotations.csv', batch_size=8, image_size=(224, 224), max_frames=10)
#     val_dataset = load_data('val_annotations.csv', batch_size=8, image_size=(224, 224), max_frames=10)
    
#     # Preprocess data
#     X_train, y_train = next(iter(train_dataset))
#     X_val, y_val = next(iter(val_dataset))
    
#     X_train, y_train = preprocess_data(X_train, y_train)
#     X_val, y_val = preprocess_data(X_val, y_val)
    
#     # Get class weights
#     class_weights = get_class_weights(y_train)
#     print("Class weights:", class_weights)
    
#     # Create data generators
#     train_datagen = ImageDataGenerator(rotation_range=15,
#                                        width_shift_range=0.1,
#                                        height_shift_range=0.1,
#                                        horizontal_flip=True)
    
#     val_datagen = ImageDataGenerator()
    
#     train_generator = train_datagen.flow(X_train, y_train, batch_size=8)
#     val_generator = val_datagen.flow(X_val, y_val, batch_size=8)
    
#     # Print shapes
#     print("Train shape:", X_train.shape, y_train.shape)
#     print("Validation shape:", X_val.shape, y_val.shape)
