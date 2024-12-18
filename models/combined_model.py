from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,TimeDistributed, MaxPooling2D, Flatten, Concatenate
from .gru import create_gru_model
from .efficientnet import create_efficientnet_base

def create_combined_model(num_frames, frame_height, frame_width, channels):
    video_input = Input(shape=(num_frames, frame_height, frame_width, channels), name='video_input')
    
    # Base CNN (EfficientNetB0)
    base_cnn_inputs, base_cnn_output = create_efficientnet_base((frame_height, frame_width, channels))
    
    # Feature extraction for each frame
    encoded_frames = TimeDistributed(base_cnn_inputs)(video_input)
    
    # Multi-scale feature extraction
    def multi_scale_features(x):
        x_small = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
        x_small = TimeDistributed(Flatten())(x_small)
        
        x_large = TimeDistributed(MaxPooling2D(pool_size=(4, 4)))(x)
        x_large = TimeDistributed(Flatten())(x_large)
        
        x = TimeDistributed(Flatten())(x)
        
        x_combined = Concatenate()([x, x_small, x_large])
        return x_combined
    
    encoded_frames = multi_scale_features(encoded_frames)
    
    # GRU for temporal dependencies
    gru_inputs, gru_outputs = create_gru_model(num_frames, encoded_frames.shape[-1])
    
    # Combine models
    combined_output = gru_outputs(encoded_frames)
    
    model = Model(inputs=video_input, outputs=combined_output, name='EfficientNetB0_GRU_Hybrid')
    
    return model
