import polars as pl
import os
import numpy as np
from sqlalchemy import create_engine
import tensorflow as tf

engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
conn = engine.connect()
ratings = pl.read_database(query='SELECT * FROM data_lake.ratings', connection=conn)
movies = pl.read_database(query='SELECT * FROM raw.movies', connection=conn)
conn.close()


    #PROSECAN BROJ OCENA PO FILMU
num_ratings = ratings.group_by('movieid').agg(pl.len().alias('#ratings_film'))
user = ratings.join(num_ratings, on = 'movieid', how = 'inner').sort(['movieid', 'userid'])
# user = user.filter(pl.col('userid').is_in(103))
movies = movies.with_columns(pl.col("genres").str.split("|"))
unique_genres = sorted(set(g for genre in movies["genres"] for g in genre))
#LAZY!
user = user.lazy()
movies = movies.lazy()
#SVI ZANROVI
for genre in unique_genres:
    movies = movies.with_columns(pl.col("genres").list.contains(genre).cast(pl.Int8).alias(genre))
movies = movies.drop('genres')
#KOLONA GODINA
movies = movies.with_columns(pl.col("title").str.extract(r"\((\d{4})\)", 1).cast(pl.Int16).alias("year"))

#ISTI FORMAT TABELE KAO MOVIES
user_zanr_train = user.join(movies, on='movieid', how='inner')

#PIVOT LONGER --> ZANROVE PREBACUJEM U JEDNU KOLONU
user_longer = (user_zanr_train.unpivot(index=['userid', 'rating'],
                                        on=unique_genres).filter(pl.col('value') == 1).rename({'variable': 'genre', 'value': 'is_genre'}))


#RACUNAM PROSEK ZA SVAKOG USERA ZA SVAKI ZANR I VRACAM U WIDE FORMAT
user_feature = user_longer.group_by('userid').agg([(pl.when(pl.col('genre') == genre).then(pl.col('rating')).mean().alias(genre)) for genre in unique_genres]).fill_null(0)
movie_avg_rating = (user.group_by('movieid').agg(pl.col('rating').mean().alias('avg_rating')))
movie_features = movies.join(movie_avg_rating, on='movieid', how='left').fill_null(0)
movie_features = movie_features.select(['movieid', 'title','year','avg_rating', *unique_genres])
df = user.join(user_feature, on="userid", how="inner").join(movie_features, on="movieid", how="inner")
df = df.collect()
movie_features = movie_features.rename({"(no genres listed)": "no genres listed"})
user_feature = user_feature.rename({"(no genres listed)": "no genres listed"})
df = df.rename({"(no genres listed)": "no genres listed"})
user_feature = user_feature.sort('userid')

df = df.sort('userid')
user = user_feature.collect()
movies = movie_features.collect()
user_id = 103

df.filter(pl.col('userid') == 103)



//
def scale(df, user, movies, user_id = None):
    y = tf.convert_to_tensor(df.select(pl.col('rating')).to_numpy(), dtype=tf.float16)
    check = df.select(pl.col(['userid', 'movieid', 'rating']))


    prva_user = df.columns.index('no genres listed')
    poslednja_user = df.columns.index('Western')
    ###prva kolona u X_user_ud je userid!!!, trebace za preporuke, za treniranje koristiti X_user
    X_user_id = tf.convert_to_tensor(df.select(['userid'] + df.columns[prva_user : poslednja_user + 1]).to_numpy(), dtype=tf.float32)
    X_movie_df = df.select(['year','avg_rating', '#ratings_film'] + [col for col in df.columns if col.endswith('_right')])
    movie_num = tf.convert_to_tensor(X_movie_df.select(['#ratings_film', 'year', 'avg_rating']).to_numpy(), dtype=tf.float32)
    movie_cat = tf.convert_to_tensor(X_movie_df.select(pl.all().exclude(['#ratings_film', 'year', 'avg_rating'])).to_numpy(), dtype=tf.float32)
    # Standardizacija user i movie numeričkih
    X_user = X_user_id[:, 1:]
    user_mean = tf.reduce_mean(X_user, axis=0)
    user_std = tf.math.reduce_std(X_user, axis=0)
    X_user_scaled = (X_user - user_mean) / (user_std)
    X_user_id_scaled = tf.concat([X_user_id[:, :1], X_user_scaled], axis=1)  # Skalirano sa ID kolonom
    movie_mean = tf.reduce_mean(movie_num, axis=0)
    movie_std = tf.math.reduce_std(movie_num, axis=0)
    movie_num_scaled = (movie_num - movie_mean) / (movie_std)
    X_movie_scaled = tf.concat([movie_num_scaled, movie_cat], axis=1)
    # Target skaliranje na [-1, 1]
    y_scaled = 2 * (y - tf.reduce_min(y)) / (tf.reduce_max(y) - tf.reduce_min(y)) - 1
    scalers = {"user_mean": user_mean, "user_std": user_std,"movie_mean": movie_mean,"movie_std": movie_std, "y_min": tf.reduce_min(y), "y_max": tf.reduce_max(y)}
    if user_id is not None:
        maska = tf.reduce_any(tf.equal(tf.expand_dims(X_user_id_scaled[:, 0], 1), tf.constant(user_id, dtype=X_user_id_scaled.dtype)), axis=1)
        X_user_id_scaled = tf.boolean_mask(X_user_id_scaled, maska)
        X_movie_scaled = tf.boolean_mask(X_movie_scaled, maska)
        y_scaled = tf.boolean_mask(y_scaled, maska)
        return X_user_id_scaled, X_movie_scaled, y_scaled, scalers
    else:
        return X_user_scaled, X_movie_scaled, y_scaled, scalers

//
user_id = [103, 104, 105, 106, 107, 108, 109, 110]
user_id = 300
tf.reduce_any(tf.equal(tf.expand_dims(X_user_id_scaled[:, 0], 1), tf.constant(user_id, dtype=X_user_id_scaled.dtype)), axis=1)
//
def scale__(df, user, movies):
    y = df.select(pl.col('rating')).to_numpy()
    prva_user = df.columns.index('no genres listed')
    poslednja_user = df.columns.index('Western')
    X_user = df.select(df.columns[prva_user : poslednja_user + 1]).to_numpy()
    X_movie_df = df.select(['year','avg_rating', '#ratings_film'] + [col for col in df.columns if col.endswith('_right')])
    movie_num = X_movie_df.select(['#ratings_film', 'year', 'avg_rating']).to_numpy()
    movie_cat = X_movie_df.select(pl.all().exclude(['#ratings_film', 'year', 'avg_rating'])).to_numpy()
    # Standardizacija user i movie numeričkih
    user_mean = X_user.mean(axis = 0)
    user_std = X_user.std(axis = 0)
    X_user_scaled = (X_user - user_mean) / (user_std)
    movie_mean = movie_num.mean(axis = 0)
    movie_std = movie_num.std(axis = 0)
    movie_num_scaled = (movie_num - movie_mean) / (movie_std)
    X_movie_scaled = np.concat([movie_num_scaled, movie_cat], axis=1)
    # Target skaliranje na [-1, 1]
    y_scaled = 2 * (y - y.min()) / (y.max() - y.min()) - 1
    scalers = {"user_mean": user_mean, "user_std": user_std, "movie_mean": movie_mean, "movie_std": movie_std, "y_min": y.min(), "y_max": y.max()}
    return X_user_scaled, X_movie_scaled, y_scaled, scalers
    
movie_num.mean().row(0)



//
def NN_prep(df, user, movies, user_id = None):
    # Priprema podataka za model
    y = tf.convert_to_tensor(df.select(pl.col('rating')).to_numpy(), dtype=tf.float16)
    prva_user = df.columns.index('no genres listed')
    poslednja_user = df.columns.index('Western')
    if user_id is None:
        X_user = tf.convert_to_tensor(df.select(df.columns[prva_user : poslednja_user + 1])
.to_numpy(), dtype=tf.float32)
    else:
        X_user = tf.convert_to_tensor(df.filter(pl.col('userid') == user_id).select(df.columns[prva_user : poslednja_user + 1]).to_numpy(), dtype=tf.float32)
    X_movie_df = df.select(['year','avg_rating', '#ratings_film'] + [col for col in df.columns if col.endswith('_right')])
    movie_num = tf.convert_to_tensor(X_movie_df.select(['#ratings_film', 'year', 'avg_rating']).to_numpy(), dtype=tf.float32)
    movie_cat = tf.convert_to_tensor(X_movie_df.select(pl.all().exclude(['#ratings_film', 'year', 'avg_rating'])).to_numpy(), dtype=tf.float32)
    # Standardizacija user i movie numeričkih 
    user_mean = tf.reduce_mean(X_user, axis=0)
    user_std = tf.math.reduce_std(X_user, axis=0)
    X_user_scaled = (X_user - user_mean) / (user_std + 1e-7)
    movie_mean = tf.reduce_mean(movie_num, axis=0)
    movie_std = tf.math.reduce_std(movie_num, axis=0)
    movie_num_scaled = (movie_num - movie_mean) / (movie_std)
    X_movie_scaled = tf.concat([movie_num_scaled, movie_cat], axis=1)
    # Target skaliranje na [-1, 1]
    y_min = tf.reduce_min(y)
    y_max = tf.reduce_max(y)
    y_scaled = 2 * (y - y_min) / (y_max - y_min) - 1

    # Vrati i transformatore za kasniju upotrebu
    scalers = {"user_mean": user_mean, "user_std": user_std, "movie_mean": movie_mean, "movie_std": movie_std, "y_min": y_min, "y_max": y_max}

    return X_user_scaled, X_movie_scaled, y_scaled, scalers

def inverse_transform_y(y_scaled, scalers):
    """
    Inverzna transformacija za y skaliran na [-1, 1].
    """
    y_min = scalers["y_min"]
    y_max = scalers["y_max"]
    y = (y_scaled + 1) * (y_max - y_min) / 2 + y_min
    return y

def inverse_transform_X_user(X_user_scaled, scalers):
    """
    Inverzna transformacija za X_user.
    """
    user_mean = scalers["user_mean"]
    user_std = scalers["user_std"]
    return X_user_scaled * (user_std + 1e-8) + user_mean

def inverse_transform_X_movie_num(X_movie_num_scaled, scalers):
    """
    Inverzna transformacija za numeričke karakteristike filma.
    """
    movie_mean = scalers["movie_mean"]
    movie_std = scalers["movie_std"]
    return X_movie_num_scaled * (movie_std + 1e-8) + movie_mean

# def batch_generator(movies, batch_size=4096):
#     engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
#     conn = engine.connect()
#     offset = 0
#     while True:
#         query = f"SELECT * FROM data_lake.ratings LIMIT {batch_size} OFFSET {offset}"
#         batch = pl.read_database(query=query, connection=conn)
#         if batch.height == 0:
#             break
#         user, movies_feat, df = prep_pipeline(batch, movies, batch)
#         X_user, X_movie, y = NN_prep(df, user, movies_feat)
#         yield (X_user, X_movie), y
#         offset += batch_size
#     conn.close()
    
    
    class ColaborativeFiltering(tf.keras.Model):
    def __init__(self, num_user_features, num_movie_features, user_layers=[128, 64], movie_layers=[128, 64], embedding=32, learning_rate=0.001, **kwargs):
        super().__init__(**kwargs)
        self.num_user_features = num_user_features
        self.num_movie_features = num_movie_features
        self.embedding = embedding
        self.learning_rate = learning_rate

        # User branch
        user_dense_layers = []
        for units in user_layers:
            user_dense_layers.append(layers.Dense(units, activation='tanh', kernel_initializer='glorot_uniform'))
        user_dense_layers.append(layers.Dense(self.embedding, activation='tanh', kernel_initializer='glorot_uniform'))
        self.user_net = tf.keras.Sequential(user_dense_layers)

        # movie branch
        movie_dense_layers = []
        for units in movie_layers:
            movie_dense_layers.append(layers.Dense(units, activation='tanh', kernel_initializer='glorot_uniform'))
        movie_dense_layers.append(layers.Dense(self.embedding, activation='tanh', kernel_initializer='glorot_uniform'))
        self.movie_net = tf.keras.Sequential(movie_dense_layers)

        self.dot = layers.Dot(axes=1, name='cosine_similarity')

        self.compile(
            optimizer=optimizers.Nadam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )

    def call(self, inputs):
        user_input, movie_input = inputs
        user_embedding = tf.nn.l2_normalize(self.user_net(user_input), axis=1)
        movie_embedding = tf.nn.l2_normalize(self.movie_net(movie_input), axis=1)
        cos_sim = self.dot([user_embedding, movie_embedding])