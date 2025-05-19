import polars as pl
from sqlalchemy import create_engine
import os
from sqlalchemy.sql import text
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


import numpy as np
np.set_printoptions(suppress=True)
import joblib

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time

engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
conn = engine.connect()

# ratings = pl.read_database(query='SELECT * FROM raw.ratings ORDER BY RANDOM() LIMIT 2500000', connection=conn)
# movies = pl.read_database(query='SELECT * FROM raw.movies', connection=conn)
# conn.close()

# conn = engine.connect()
# ratings.write_database(
#     table_name="data_lake.ratings",
#     connection=conn,
#     if_table_exists="replace",
# )

# print("Uspesno upisani redovi")
# conn.close()

start_time = time.time()
engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
conn = engine.connect()
ratings = pl.read_database(query='SELECT * FROM data_lake.ratings', connection=conn)
movies = pl.read_database(query='SELECT * FROM raw.movies', connection=conn)
print(f"Execution time: {time.time() - start_time:.4f} seconds")
ratings.describe()[['statistic', 'rating']]
movies

def prep_pipeline(user, movies):
    #PROSECAN BROJ OCENA PO FILMU
    num_ratings = ratings.group_by('movieid').agg(pl.len().alias('#ratings_film')).filter(pl.col('#ratings_film') > 40)
    user = ratings.join(num_ratings, on = 'movieid', how = 'inner').sort(['movieid', 'userid'])
    movies = movies.with_columns(pl.col("genres").str.split("|"))
    unique_genres = sorted(set(g for genre_list in movies["genres"] for g in genre_list))
    #LAZY!
    user = user.lazy()
    movies = movies.lazy()
    #SVI ZANROVI
    for genre in unique_genres:
        movies = movies.with_columns(pl.col("genres").list.contains(genre).cast(pl.Int8).alias(genre))
    movies = movies.drop('genres')
    #KOLONA GODINA
    movies = movies.with_columns(pl.col("title").str.extract(r"\((\d{4})\)", 1).cast(pl.Int16).alias("year"))
    movies =movies.select(['movieid', 'title', 'year', *unique_genres])
    
    #ISTI FORMAT TABELE KAO MOVIES
    user_zanr_train = user.join(movies, on='movieid', how='inner')
    
    #PIVOT LONGER --> ZANROVE PREBACUJEM U JEDNU KOLONU
    user_longer = (user_zanr_train.unpivot(index=['userid', 'rating'],
                                           on=unique_genres).filter(pl.col('value') == 1).rename({'variable': 'genre', 'value': 'is_genre'}))
    
    #RACUNAM PROSEK ZA SVAKOG USERA ZA SVAKI ZANR I VRACAM U WIDE FORMAT
    user_feature = user_longer.group_by('userid').agg([(pl.when(pl.col('genre') == genre).then(pl.col('rating'))
                                                        .otherwise(None).mean().alias(genre)) for genre in unique_genres]).fill_null(0)
    movie_avg_rating = (user.group_by('movieid').agg(pl.col('rating').mean().alias('avg_rating')))
    movie_features = movies.join(movie_avg_rating, on='movieid', how='left').fill_null(0)
    movie_features = movie_features.select(['movieid', 'title','year','avg_rating', *unique_genres])
    df = user.join(user_feature, on="userid", how="inner").join(movie_features, on="movieid", how="inner")
    df = df.collect()
    movie_features = movie_features.rename({"(no genres listed)": "no genres listed"})
    user_feature = user_feature.rename({"(no genres listed)": "no genres listed"})
    df = df.rename({"(no genres listed)": "no genres listed"})

    return user_feature.collect(), movie_features.collect(), df
user, movies, df = prep_pipeline(ratings, movies)

def NN_prep(df):
    # Priprema podataka bez sklearn i numpy
    y = tf.convert_to_tensor(df.select(pl.col('rating')).to_series().to_list(), dtype=tf.float32)
    X_user = tf.convert_to_tensor(df.select(user.select(pl.exclude('userid')).columns).to_numpy(), dtype=tf.float32)
    X_movie_df = df.select(movies.select(pl.exclude(['movieid','title'])).columns + ['#ratings_film'])
    movie_num = tf.convert_to_tensor(X_movie_df.select(['#ratings_film', 'year', 'avg_rating']).to_numpy(), dtype=tf.float32)
    movie_cat = tf.convert_to_tensor(X_movie_df.select(pl.all().exclude(['#ratings_film', 'year', 'avg_rating'])).to_numpy(), dtype=tf.float32)
    # Skaliranje (standardizacija) user i movie numeričkih karakteristika
    user_mean = tf.reduce_mean(X_user, axis=0)
    user_std = tf.math.reduce_std(X_user, axis=0)
    X_user = (X_user - user_mean) / (user_std + 1e-8)
    movie_mean = tf.reduce_mean(movie_num, axis=0)
    movie_std = tf.math.reduce_std(movie_num, axis=0)
    movie_num_scaled = (movie_num - movie_mean) / (movie_std + 1e-8)
    X_movie = tf.concat([movie_num_scaled, movie_cat], axis=1)
    # Target skaliranje na [-1, 1]
    y_min = tf.reduce_min(y)
    y_max = tf.reduce_max(y)
    y_scaled = 2 * (y - y_min) / (y_max - y_min) - 1
    return X_user, X_movie, y_scaled
X_user_numpy, X_movie_numpy, y = NN_prep(df)
tf.reduce_max(y), tf.reduce_min(y) 
X_user_numpy.shape, X_movie_numpy.shape, y.shape, SS_movie.feature_names_in_, SS_user.feature_names_in_,SS_target.feature_range

##Za linearnu regresiju
X = np.column_stack([X_user_numpy, X_movie_numpy])
X.shape, y.shape

reg = LinearRegression()
reg.fit(X, y.reshape(-1,))
pred = reg.predict(X)
mean_squared_error(pred, y), mean_absolute_error(pred, y)

num_user_features = X_user_numpy.shape[1]
num_item_features = X_movie_numpy.shape[1]
num_outputs = 20

X_user_train, X_user_dev, X_movie_train, X_movie_dev, y_train, y_dev = train_test_split(X_user_numpy, X_movie_numpy, y, test_size=0.15, random_state=42)
X_user_train.shape, X_user_dev.shape, X_movie_train.shape, X_movie_dev.shape, y_train.shape, y_dev.shape
X_user_train[:5], X_movie_train[:5], y_train[:5]

num_user_features = X_user_numpy.shape[1]
num_item_features = X_movie_numpy.shape[1]
num_outputs = 64

class ColaborativeFiltering:
    def __init__(self, num_user_features, num_item_features, embedding=64, learning_rate=0.001):
        self.num_user_features = num_user_features
        self.num_item_features = num_item_features
        self.embedding = embedding
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        user_input = keras.Input(shape=(self.num_user_features,), name='user_input')
        x_user = layers.Dense(128, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=keras.regularizers.l2(0.001))(user_input)
        x_user = layers.Dense(self.embedding, activation='tanh', kernel_initializer='glorot_uniform')(x_user)
        user_embedding = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(x_user)

        item_input = keras.Input(shape=(self.num_item_features,), name='item_input')
        x_item = layers.Dense(128, activation='tanh', kernel_initializer='glorot_uniform')(item_input)
        x_item = layers.Dense(self.embedding, activation='tanh', kernel_initializer='glorot_uniform')(x_item)
        item_embedding = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(x_item)

        cos_sim = layers.Dot(axes=1, name='cosine_similarity')([user_embedding, item_embedding])
        model = keras.Model(inputs=[user_input, item_input], outputs=cos_sim)
        model.compile(optimizer=keras.optimizers.Nadam(learning_rate=self.learning_rate), loss='mse', metrics=['mae', 'mse'])
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

# Primer korišćenja klase:
model = ColaborativeFiltering(num_user_features, num_item_features, embedding=64, learning_rate=0.001)
model.summary()
callbacks = [EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
             ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6, verbose=1)]
history = model.fit(X_user_train, X_movie_train, y_train, X_user_dev, X_movie_dev, y_dev, epochs=25, batch_size=258, callbacks=callbacks)
model.save('model.keras')
y_pred = model.predict(X_user_dev, X_movie_dev)

# Primer: preporuke za jednog usera
user_idx = 0
user_id = df['userid'][user_idx] if 'userid' in df.columns else user_idx
user_seen_movie_indices = model.get_user_seen_movie_indices(user_id, ratings, movies)
user_vec = X_user_dev[user_idx]
k = 10
movie_titles = None  # ili lista naslova
preporuke = model.recommend(user_vec, X_movie_numpy, user_seen_movie_indices, k=k, movie_titles=movie_titles)
print('Preporuke za usera', user_idx, ':', preporuke)

joblib.dump(SS_movie, 'SS_movie.pkl')
joblib.dump(SS_user, 'SS_user.pkl') 
joblib.dump(SS_target, 'SS_target.pkl')
joblib.dump(history, 'history.pkl')

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
def batch_generator(batch_size=4096):
    offset = 0
    while True:
        query = f"SELECT * FROM raw.ratings LIMIT {batch_size} OFFSET {offset}"
        batch = pl.read_database(query=query, connection=conn)
        if batch.height == 0:
            break
        # Pripremi batch podatke (pretpostavlja se da su movies i user_feature globalni ili dostupni)
        # Ako treba, koristi prep_pipeline za feature engineering
        # Ovde koristiš istu logiku kao u NN_prep, ali za batch
        # Ako treba, možeš unapred izračunati mean/std i proslediti ih
        X_user, X_movie, y = NN_prep(batch)
        yield (X_user, X_movie), y
        offset += batch_size


dataset = tf.data.Dataset.from_generator(
    lambda: batch_generator(batch_size=4096),
    output_signature=(
        (tf.TensorSpec(shape=(None, X_user_numpy.shape[1]), dtype=tf.float32),
         tf.TensorSpec(shape=(None, X_movie_numpy.shape[1]), dtype=tf.float32)),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
)
dataset.prefetch(tf.data.AUTOTUNE)
# Sada možeš trenirdaati model sa:
# model.fit(dataset, epochs=10, steps_per_epoch=broj_redova // batch_size)

conn.execute("SELECT COUNT(*) FROM data_lake.ratings")
conn.execute(text("SELECT COUNT(*) FROM raw.ratings")).scalar()
conn.execute(text("SELECT COUNT(*) FROM data_lake.ratings")).scalar()