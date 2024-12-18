import numpy as np
import tensorflow.keras.backend as K

def compute_class_weights(y_int):
    class_weights = np.array([
        len(y_int) / (len(np.unique(y_int)) * np.sum(y_int == t))
        for t in np.unique(y_int)
    ])
    return class_weights

def weighted_categorical_crossentropy(class_weights_tensor):
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, y_pred.dtype)
        weights = y_true * class_weights_tensor
        weights = K.sum(weights, axis=-1)
        cce = K.categorical_crossentropy(y_true, y_pred)
        return cce * weights
    return loss
