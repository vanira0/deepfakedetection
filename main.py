import os
import json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from utils.data_loader import create_train_val_loaders
from models.combined_model import create_combined_model
from utils.metrics import evaluate_model

def main():
    # Load configuration
    with open('experiments/config.json') as config_file:
        config = json.load(config_file)

    # Set up experiment directory
    exp_dir = f'experiments/{config["experiment_name"]}'
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Experiment directory created: {exp_dir}")

    # Create data loaders
    train_loader, val_loader = create_train_val_loaders(
        train_csv=config['dataset']['metadata_dir'] + 'train_annotations.csv',
        val_csv=config['dataset']['metadata_dir'] + 'val_annotations.csv',
        batch_size=config['training']['batch_size'],
        image_size=tuple(config['preprocessing']['image_size']),
        max_frames=config['preprocessing']['max_frames']
    )

    # Initialize model
    # model = CombinedModel(input_shape=(config['preprocessing']['max_frames'], *config['preprocessing']['image_size'], 3),
    #                       num_classes=2,
    #                       efficientnet_variant=config['model']['efficientnet_variant'],
    #                       gru_units=config['model']['gru_units'],
    #                       dense_layers=config['model']['dense_layers'])
    num_frames = config['preprocessing']['max_frames']
    frame_height, frame_width = config['preprocessing']['image_size']
    channels = 3
    
    model = create_combined_model(num_frames, frame_height, frame_width, channels)

    # Compile model
    optimizer = Adam(learning_rate=config['training']['learning_rate'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=config['callbacks']['early_stopping_patience'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=config['callbacks']['reduce_lr_factor'],
                                  patience=config['callbacks']['reduce_lr_patience'], min_delta=0.001, verbose=1)

    # Train the model
    history = model.fit(train_loader, 
                         validation_data=val_loader, 
                         epochs=config['training']['epochs'],
                         callbacks=[early_stopping, reduce_lr],
                         verbose=1)

    # Save the best model
    model.save(os.path.join(exp_dir, 'best_model.h5'))

    # Evaluate the best model
    y_true = []
    y_pred = []

    for X_batch, y_batch in val_loader:
        y_true.extend(np.argmax(y_batch, axis=-1))
        y_pred.extend(model.predict(X_batch))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    evaluation_results = evaluate_model(y_true, y_pred)

    print("Evaluation Results:")
    for metric, value in evaluation_results.items():
        if isinstance(value, dict):
            print(f"\n{metric}:")
            for sub_metric, sub_value in value.items():
                print(f"  {sub_metric}: {sub_value}")
        else:
            print(f"{metric}: {value}")

    # Save evaluation results
    with open(os.path.join(exp_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    print(f"Experiment completed. Results saved in {exp_dir}")

if __name__ == "__main__":
    main()



























# from utils.data_loader import create_train_val_loaders, preprocess_data, get_class_weights
# from utils.data_preparation import prepare_dataset

# # Prepare the dataset
# prepare_dataset()

# # Create data loaders
# train_loader, val_loader = create_train_val_loaders(
#     train_csv='path/to/train_annotations.csv',
#     val_csv='path/to/val_annotations.csv',
#     batch_size=8,
#     image_size=(224, 224),
#     max_frames=10
# )

# # Preprocess data
# X_train, y_train = next(iter(train_loader))
# X_val, y_val = next(iter(val_loader))

# X_train, y_train = preprocess_data(X_train, y_train)
# X_val, y_val = preprocess_data(X_val, y_val)

# # Get class weights
# class_weights = get_class_weights(y_train)
# print("Class weights:", class_weights)

