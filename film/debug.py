import polars as pl
import os
from sqlalchemy import create_engine
import tensorflow as tf
from tqdm import tqdm 
import numpy as np


def prep_pipeline(ratings, movies):
    '''
    Priprema za model
    '''
    #PROSECAN BROJ OCENA PO FILMU
    num_ratings = ratings.group_by('movieid').agg(pl.len().alias('#ratings_film'))
    user = ratings.join(num_ratings, on = 'movieid', how = 'inner').sort(['movieid', 'userid'])
    movies, unique_genres = get_genres(movies, prep = True)
    #LAZY!
    user = user.lazy()
    movies = movies.lazy()
    ratings = ratings.lazy()
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

    X_user = ratings.join(user_feature, on="userid", how="inner").drop('rating').collect()

    X_movie = user.join(movie_features, on="movieid", how="inner").drop('rating', 'title').collect()
    
    return X_user, X_movie

def get_genres(movies, prep = False):
    movies = movies.with_columns(pl.col("genres").str.split("|"))
    unique_genres = sorted(set(g for genre in movies["genres"] for g in genre))
    unique_genres[0] = unique_genres[0].replace('(', '').replace(')', '')
    if prep == True:
      return movies, unique_genres
    else:
      return unique_genres
  
def batch_generator(movies, batch_size=1000000, total = 2e7):
    '''
    Pravi skupove od batch_size (milion) iz nasumicnih total (20 miliona) redova u tabeli ratings
    '''
    engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
    conn = engine.connect()
    offset = 0
    while offset < total:
        query = f"SELECT * FROM raw.ratings LIMIT {batch_size} OFFSET {offset}"
        batch = pl.read_database(query=query, connection=conn)
        if batch.height == 0:
            break
        user, movie = prep_pipeline(batch, movies)
        
        # X_user, X_movie, y = scale(df, user, movies_feat)
        # yield (X_user, X_movie), tf.squeeze(y)
        user = tf.convert_to_tensor(user.to_numpy(), dtype=tf.float32)
        movie = tf.convert_to_tensor(movie.to_numpy(), dtype=tf.float32)
        yield (user, movie)
        offset += batch_size
    conn.close()

engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
conn = engine.connect()
movies = pl.read_database(query='SELECT * FROM raw.movies', connection=conn)
conn.close()

batch_size=1000000
total=20000000
from itertools import islice
data = tf.data.Dataset.from_generator(lambda: batch_generator(movies, batch_size=batch_size, total=total), output_signature= (
    tf.TensorSpec(shape=(None, 22), dtype=tf.float32, name = 'user'),  # User features
    tf.TensorSpec(shape=(None, 25), dtype=tf.float32, name = 'movie')   # Movie features
))
#samo za RZS
movies = pl.read_csv(r'https://raw.githubusercontent.com/BogdanSliskovic/ML/refs/heads/main/film/movies.csv')
ratings = pl.read_csv(r'https://raw.githubusercontent.com/BogdanSliskovic/ML/refs/heads/main/film/ratings_RZS.csv')
X_user, X_movie = prep_pipeline(ratings, movies)
X_user[:,2].mean()

n = 0
def get_mean(data):
    n = 0
    user_mean = tf.zeros(20)
    movie_mean = tf.zeros(3)
    for user, movie in (tqdm(data)):
        user_num = user[:, 2:] 
        movie_num = movie[:, 2:5] 
        for indeks, feat in enumerate([user_num, movie_num]):
            maska = feat > 0
            mask_float = tf.cast(maska, tf.float32)
            sum_per_col = tf.reduce_sum(tf.where(maska, feat, tf.zeros_like(feat)), axis=0)
            count_per_col = tf.reduce_sum(mask_float, axis=0)
            mean_per_col = tf.where(count_per_col > 0, sum_per_col / count_per_col , tf.zeros_like(count_per_col))

            if indeks == 0:
                user_mean += mean_per_col
            else:
                movie_mean += mean_per_col

        n += 1
    user_mean /= n
    movie_mean /= n
    return user_mean, movie_mean

user_mean, movie_mean = get_mean(data)



def scale(df, user, movies, user_id = None):
    '''
    Skaliranje numeričkih karakteristika i prebacivanje u tenzore
    df - Polars DataFrame sa svim podacima
    user - Polars DataFrame sa korisničkim karakteristikama
    movies - Polars DataFrame sa filmskim karakteristikama
    user_id - ako je None, onda se vracaju svi korisnici, ako je lista (ili int) onda se vraca samo taj korisnik
    
    '''
    y = tf.convert_to_tensor(df.select(pl.col('rating')).to_numpy(), dtype=tf.float16)

    prva_user = df.columns.index('no genres listed')
    poslednja_user = df.columns.index('Western')
    ###prva kolona u X_user_ud je userid!!!, trebace za preporuke, za treniranje koristiti X_user
    X_user = tf.convert_to_tensor(df.select(['userid'] + df.columns[prva_user : poslednja_user + 1]).to_numpy(), dtype=tf.float32)
    X_movie_df = df.select(['year','avg_rating', '#ratings_film'] + [col for col in df.columns if col.endswith('_right')])
    movie_num = tf.convert_to_tensor(X_movie_df.select(['#ratings_film', 'year', 'avg_rating']).to_numpy(), dtype=tf.float32)
    movie_cat = tf.convert_to_tensor(X_movie_df.select(pl.all().exclude(['#ratings_film', 'year', 'avg_rating'])).to_numpy(), dtype=tf.float32)
    # Standardizacija user i movie numeričkih
 
    return X_user, X_movie_scaled, y_scaled#, scalers

def 