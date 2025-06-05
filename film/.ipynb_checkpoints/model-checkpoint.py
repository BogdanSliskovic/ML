import numpy as np
import tensorflow as tf
from keras import layers, regularizers, optimizers
import polars as pl
from keras.saving import register_keras_serializable

@register_keras_serializable()
class ColaborativeFiltering(tf.keras.Model):
    def __init__(self, num_user_features, num_movie_features, user_layers=[128, 64], movie_layers=[128, 64], embedding=32, learning_rate=0.001, user_reg = None, movie_reg = None, **kwargs):
        super().__init__(**kwargs)
        self.num_user_features = num_user_features
        self.num_movie_features = num_movie_features
        self.embedding = embedding
        self.learning_rate = learning_rate

        user_dense_layers = []
        for i, units in enumerate(user_layers):
            reg = user_reg[i] if user_reg is not None else None
            user_dense_layers.append(layers.Dense(units, activation='relu', kernel_initializer='he_normal', kernel_regularizer=reg))

        user_dense_layers.append(layers.Dense(self.embedding, activation='tanh', kernel_initializer='he_normal'))
        self.user_net = tf.keras.Sequential(user_dense_layers)

        movie_dense_layers = []
        for units in movie_layers:
            movie_dense_layers.append(layers.Dense(units, activation='tanh', kernel_initializer='he_normal'))
        movie_dense_layers.append(layers.Dense(self.embedding, activation='tanh', kernel_initializer='he_normal'))
        self.movie_net = tf.keras.Sequential(movie_dense_layers)

        self.dot = layers.Dot(axes=1, name='cosine_similarity')

        # Save architecture parameters for serialization
        self.user_layers = user_layers
        self.movie_layers = movie_layers
        self.user_reg = user_reg
        self.movie_reg = movie_reg

        self.compile(
            optimizer=optimizers.Nadam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_user_features': self.num_user_features,
            'num_movie_features': self.num_movie_features,
            'embedding': self.embedding,
            'learning_rate': self.learning_rate,
            'user_layers': self.user_layers,
            'movie_layers': self.movie_layers,
            'user_reg': self.user_reg,
            'movie_reg': self.movie_reg
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs):
        user_input, movie_input = inputs
        user_embedding = tf.nn.l2_normalize(self.user_net(user_input), axis=1)
        movie_embedding = tf.nn.l2_normalize(self.movie_net(movie_input), axis=1)
        cos_sim = self.dot([user_embedding, movie_embedding])
        return cos_sim

    def recommend(self, user_vec, movie_matrix, user_seen_movie_indices = None, k=10, movie_titles=None):
        user_vecs = tf.repeat(tf.reshape(user_vec, (1, -1)), tf.shape(movie_matrix)[0], axis=0)
        preds = self.predict([user_vecs, movie_matrix])
        # mask_indices = tf.constant(list(user_seen_movie_indices), dtype=tf.int32)
        # preds = tf.tensor_scatter_nd_update(
        #     tf.squeeze(preds),
        #     tf.expand_dims(mask_indices, 1),
        #     tf.fill([tf.size(mask_indices)], tf.constant(-float('inf'), dtype=preds.dtype))
        # )
        top_k_idx = tf.argsort(preds, direction='DESCENDING')[:k]
        if movie_titles is not None:
            return [(movie_titles[int(i)], float(preds[i])) for i in top_k_idx]
        else:
            return [(int(i), float(preds[i])) for i in top_k_idx]

    def get_user_seen_movie_indices(self, user_id, ratings, movies):
        gledani_movieid = set(ratings.filter(pl.col('userid') == user_id)['movieid'].to_list())
        movieid_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies['movieid'].to_list())}
        return {movieid_to_idx[movie_id] for movie_id in gledani_movieid if movie_id in movieid_to_idx}