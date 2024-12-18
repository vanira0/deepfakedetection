from tensorflow.keras.layers import Input,GRU, Dense, Dropout

def create_gru_model(num_frames, feature_size):
    inputs = Input(shape=(num_frames, feature_size), name='gru_input')
    
    x = GRU(units=128, return_sequences=False)(inputs)
    x = Dropout(0.3)(x)
    outputs = Dense(2, activation='softmax')(x)
    
    return inputs, outputs
