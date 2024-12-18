from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def create_efficientnet_base(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape, name='efficientnet_input')
    
    # Load pre-trained EfficientNetB0 model
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=inputs)
    
    # Freeze pre-trained layers
    base_model.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    
    return Model(inputs=inputs, outputs=x, name="EfficientNetB0_Base")
