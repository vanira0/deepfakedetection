from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, Dense, Concatenate
from .gru import create_gru_model
from .efficientnet import create_efficientnet_base

def create_combined_model(num_frames, frame_height, frame_width, channels):
    # Video input
    video_input = Input(shape=(num_frames, frame_height, frame_width, channels), name='video_input')
    
    # Base CNN (EfficientNetB0)
    efficientnet_model = create_efficientnet_base((frame_height, frame_width, channels))
    frame_features = TimeDistributed(efficientnet_model, name='time_distributed_efficientnet')(video_input)
    
    # Feature transformation (instead of spatial pooling)
    def transform_features(x):
        # Apply dense layers to transform the feature vector
        transformed = TimeDistributed(Dense(512, activation='relu', name='dense_transform'))(x)
        return transformed
    
    transformed_features = transform_features(frame_features)
    
    # GRU Model
    gru_model = create_gru_model(num_frames, transformed_features.shape[-1])
    gru_output = gru_model(transformed_features)
    
    # Combined model
    model = Model(inputs=video_input, outputs=gru_output, name='EfficientNet_GRU_Hybrid')
    
    return model
