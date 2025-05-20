import numpy as np
import tensorflow as tf
from keras import layers, Input, regularizers, Model, optimizers
from tensorflow import keras
import polars as pl

class ColaborativeFiltering:
    def __init__(self, num_user_features, num_item_features, embedding=64, learning_rate=0.001):
        self.num_user_features = num_user_features
        self.num_item_features = num_item_features
        self.embedding = embedding
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        user_input = Input(shape=(self.num_user_features,), name='user_input')
        x_user = layers.Dense(128, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.001))(user_input)
        x_user = layers.Dense(self.embedding, activation='tanh', kernel_initializer='glorot_uniform')(x_user)
        user_embedding = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1), output_shape=(self.embedding,))(x_user)

        item_input = Input(shape=(self.num_item_features,), name='item_input')
        x_item = layers.Dense(128, activation='tanh', kernel_initializer='glorot_uniform')(item_input)
        x_item = layers.Dense(self.embedding, activation='tanh', kernel_initializer='glorot_uniform')(x_item)
        item_embedding = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1), output_shape=(self.embedding,))(x_item)

        cos_sim = layers.Dot(axes=1, name='cosine_similarity')([user_embedding, item_embedding])
        model = Model(inputs=[user_input, item_input], outputs=cos_sim)
        model.compile(optimizer=optimizers.Nadam(learning_rate=self.learning_rate), loss='mse', metrics=['mae', 'mse'])
        return model

    def fit(self, X_user_train, X_item_train, y_train, X_user_val, X_item_val, y_val, epochs=25, batch_size=512, callbacks=None):
        return self.model.fit(
            x=[X_user_train, X_item_train], y=y_train,
            validation_data=([X_user_val, X_item_val], y_val),
            callbacks=callbacks,
            epochs=epochs, batch_size=batch_size
        )

    def predict(self, X_user, X_item):
        return self.model.predict([X_user, X_item])

    def save(self, path):
        self.model.save(path)

    def summary(self):
        self.model.summary()

    def recommend(self, user_vec, movie_matrix, user_seen_movie_indices, k=10, movie_titles=None):
        # Prosiri user_vec na broj filmova
        user_vecs = np.repeat(user_vec.reshape(1, -1), movie_matrix.shape[0], axis=0)
        preds = self.predict(user_vecs, movie_matrix).flatten()
        preds[list(user_seen_movie_indices)] = -np.inf
        top_k_idx = preds.argsort()[-k:][::-1]
        if movie_titles is not None:
            return [(movie_titles[i], preds[i]) for i in top_k_idx]
        else:
            return list(zip(top_k_idx, preds[top_k_idx]))

    def get_user_seen_movie_indices(self, user_id, ratings, movies):
        # Pronađi movieid koje je user gledao
        gledani_movieid = set(ratings.filter(pl.col('userid') == user_id)['movieid'].to_list())
        # Mapiraj movieid na indekse u X_movie_numpy
        movieid_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies['movieid'].to_list())}
        return {movieid_to_idx[movie_id] for movie_id in gledani_movieid if movie_id in movieid_to_idx}
