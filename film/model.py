import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.saving import register_keras_serializable

def scale_y(y):
    '''od -1 do 1 (0.5 je kada je ocena nula)'''
    return 2 * (y - 0.5) / 4.5 - 1

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

@register_keras_serializable()
class L2NormalizeLayer(layers.Layer):
    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=1)
@register_keras_serializable()
class SqueezeLayer(layers.Layer):
    def call(self, inputs):
        return tf.squeeze(inputs, axis=1)
    
@register_keras_serializable()
class MovieRecommender(tf.keras.Model):
    def __init__(self, user_input_dim, movie_input_dim, hidden_layers, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.user_input_dim = user_input_dim
        self.movie_input_dim = movie_input_dim
        self.hidden_layers = hidden_layers
        self.embedding_dim = embedding_dim

        # USER mreža
        self.user_net = keras.Sequential(name="user_mreza")
        for units in hidden_layers:
            self.user_net.add(layers.Dense(units, kernel_initializer='he_normal'))
            self.user_net.add(layers.BatchNormalization())
            self.user_net.add(layers.Activation('relu'))
        self.user_net.add(layers.Dense(embedding_dim, activation='relu', kernel_initializer='he_normal'))
        self.user_net.add(L2NormalizeLayer())

        # MOVIE mreža
        self.movie_net = keras.Sequential(name="movie_mreza")
        for units in hidden_layers:
            self.movie_net.add(layers.Dense(units, kernel_initializer='he_normal'))
            self.movie_net.add(layers.BatchNormalization())
            self.movie_net.add(layers.Activation('relu'))
        self.movie_net.add(layers.Dense(embedding_dim, activation='relu', kernel_initializer='he_normal'))
        self.movie_net.add(L2NormalizeLayer())

        # Izlaz
        self.dot = layers.Dot(axes=1, name='cosine_similarity')
        self.squeeze = SqueezeLayer()

    def call(self, inputs):
        user_input, movie_input = inputs
        user_embedding = self.user_net(user_input)
        movie_embedding = self.movie_net(movie_input)
        similarity = self.dot([user_embedding, movie_embedding])
        return self.squeeze(similarity)
    def get_config(self):
        config = super().get_config()
        config.update({
            "user_input_dim": self.user_input_dim,
            "movie_input_dim": self.movie_input_dim,
            "hidden_layers": self.hidden_layers,
            "embedding_dim": self.embedding_dim
        })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)