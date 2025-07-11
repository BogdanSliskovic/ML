import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import layers
from keras.saving import register_keras_serializable

@register_keras_serializable()
class StandardizationLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # axis je fiksiran na 1
        self.norm = layers.Normalization(axis=1)

    def adapt(self, data):
        self.norm.adapt(data)

    def call(self, inputs):
        return self.norm(inputs)

    def get_config(self):
        config = super().get_config()
        return config
    
def standardizacija(var):
    '''var -> dataset.map() 
    vraca adaptiran normalization sloj 
    '''
    x = layers.Normalization()
    x.adapt(var)
    return x

def scale_y(y):
    '''od -1 do 1 (0.5 je kada je ocena nula)'''
    return 2 * (y - 0.5) / 4.5 - 1

@register_keras_serializable()
class L2NormalizeLayer(layers.Layer):
    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=1)
@register_keras_serializable()
class SqueezeLayer(layers.Layer):
    def call(self, inputs):
        return tf.squeeze(inputs, axis=1)
    