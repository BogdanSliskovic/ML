import polars as pl
import os
import numpy as np
from sqlalchemy import create_engine
import tensorflow as tf

'''Funkcije za pripremu podataka za collaborative filtering model'''

def read_data_lake():
    engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
    conn = engine.connect()
    ratings = pl.read_database(query='SELECT * FROM data_lake.ratings', connection=conn)
    movies = pl.read_database(query='SELECT * FROM raw.movies', connection=conn)
    conn.close()
    return ratings, movies

def prep_pipeline(user, movies, ratings):
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



def NN_prep(df, user, movies):
    # Priprema podataka bez sklearn i numpy
    y = tf.convert_to_tensor(df.select(pl.col('rating')).to_series().to_list(), dtype=tf.float32)
    X_user = tf.convert_to_tensor(df.select(user.select(pl.exclude('userid')).columns).to_numpy(), dtype=tf.float32)
    X_movie_df = df.select(movies.select(pl.exclude(['movieid','title'])).columns + ['#ratings_film'])
    movie_num = tf.convert_to_tensor(X_movie_df.select(['#ratings_film', 'year', 'avg_rating']).to_numpy(), dtype=tf.float32)
    movie_cat = tf.convert_to_tensor(X_movie_df.select(pl.all().exclude(['#ratings_film', 'year', 'avg_rating'])).to_numpy(), dtype=tf.float32)
    # Skaliranje (standardizacija) user i movie numeričkih karakteristika
    user_mean = tf.reduce_mean(X_user, axis=0)
    user_std = tf.math.reduce_std(X_user, axis=0)
    X_user_scaled = (X_user - user_mean) / (user_std + 1e-8)
    movie_mean = tf.reduce_mean(movie_num, axis=0)
    movie_std = tf.math.reduce_std(movie_num, axis=0)
    movie_num_scaled = (movie_num - movie_mean) / (movie_std + 1e-8)
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

def batch_generator(movies, batch_size=4096):
    engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
    conn = engine.connect()
    offset = 0
    while True:
        query = f"SELECT * FROM data_lake.ratings LIMIT {batch_size} OFFSET {offset}"
        batch = pl.read_database(query=query, connection=conn)
        if batch.height == 0:
            break
        user, movies_feat, df = prep_pipeline(batch, movies, batch)
        X_user, X_movie, y = NN_prep(df, user, movies_feat)
        yield (X_user, X_movie), y
        offset += batch_size
    conn.close()
    