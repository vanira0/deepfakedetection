import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
from tensorflow.keras.utils import to_categorical
from src.models.deepfake_model import build_deepfake_model
from src.utils.data_loader import create_train_val_loaders, preprocess_data, get_class_weights
from config import BATCH_SIZE, EPOCHS, IMAGE_SIZE, MAX_FRAMES, CHECKPOINT_DIR

def train_model(model, train_loader, val_loader, class_weights=None):
    # Define paths for saving checkpoints and logs
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Model checkpointing: Save the best model based on validation loss
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_DIR, "best_model.h5"),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    # Early stopping: Stop training if no improvement in validation loss for 3 epochs
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        verbose=1,
        restore_best_weights=True
    )

    # Learning rate scheduler: Reduce learning rate if validation loss plateaus
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    # Training loop
    history = model.fit(
        train_loader,
        epochs=EPOCHS,
        validation_data=val_loader,
        callbacks=[checkpoint, early_stopping, lr_scheduler],
        class_weight=class_weights,
        verbose=1
    )

    return history

if __name__ == "__main__":
    # Load and preprocess data
    train_loader, val_loader = create_train_val_loaders(
        train_csv='path/to/train_annotations.csv',
        val_csv='path/to/val_annotations.csv',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        max_frames=MAX_FRAMES
    )

    # Get class weights
    _, y_train = next(iter(train_loader))
    class_weights = get_class_weights(y_train)

    # Build the model
    model = build_deepfake_model(input_shape=(MAX_FRAMES, *IMAGE_SIZE, 3), num_classes=2)

    # Compile the model
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Train the model
    history = train_model(model, train_loader, val_loader, class_weights)

    # Save the final model
    model.save(os.path.join(CHECKPOINT_DIR, "final_model.h5"))

    # Plot training and validation accuracy and loss
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, "training_plots.png"))
    plt.close()
