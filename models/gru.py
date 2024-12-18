from tensorflow.keras.layers import Input, GRU, Dense, Dropout
from tensorflow.keras.models import Model

def create_gru_model(num_frames, feature_size):
    inputs = Input(shape=(num_frames, feature_size), name='gru_input')
    
    x = GRU(units=128, return_sequences=False, name='gru_layer')(inputs)
    x = Dropout(0.3)(x)
    outputs = Dense(2, activation='softmax', name='output_layer')(x)
    
    return Model(inputs=inputs, outputs=outputs, name='GRU_Model')
