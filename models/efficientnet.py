from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, GlobalAveragePooling2D

def create_efficientnet_base(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape, name='input')
    
    # Load pre-trained EfficientNetB0 model
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=inputs)
    
    # Freeze pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    
    return inputs, x
