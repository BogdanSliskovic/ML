import polars as pl
import os
from sqlalchemy import create_engine
import tensorflow as tf
from tqdm import tqdm 

'''Funkcije za pripremu podataka za collaborative filtering model'''

def read_data_lake():
    '''
    Data lake --> Polars.DataFrame
    '''
    engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
    conn = engine.connect()
    ratings = pl.read_database(query='SELECT * FROM data_lake.ratings', connection=conn)
    movies = pl.read_database(query='SELECT * FROM raw.movies', connection=conn)
    conn.close()
    return ratings, movies

def prep_pipeline(ratings, movies, user_id = None):
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
    
    return user_feature.collect(), movie_features.collect(), df


def get_genres(movies, prep = False):
    movies = movies.with_columns(pl.col("genres").str.split("|"))
    unique_genres = sorted(set(g for genre in movies["genres"] for g in genre))
    unique_genres[0] = unique_genres[0].replace('(', '').replace(')', '')
    if prep == True:
      return movies, unique_genres
    else:
      return unique_genres
    
def global_scalers():
    engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
    conn = engine.connect()
    df = pl.read_database(query='SELECT * FROM raw.ratings', connection=conn)
    user, movies_feat, df = prep_pipeline(df, pl.read_database(query='SELECT * FROM raw.movies', connection=conn))
    _, _ , _, scalers = scale(df, user, movies_feat)
    conn.close()
    return scalers

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
    X_user_id = tf.convert_to_tensor(df.select(['userid'] + df.columns[prva_user : poslednja_user + 1]).to_numpy(), dtype=tf.float32)
    X_movie_df = df.select(['year','avg_rating', '#ratings_film'] + [col for col in df.columns if col.endswith('_right')])
    movie_num = tf.convert_to_tensor(X_movie_df.select(['#ratings_film', 'year', 'avg_rating']).to_numpy(), dtype=tf.float32)
    movie_cat = tf.convert_to_tensor(X_movie_df.select(pl.all().exclude(['#ratings_film', 'year', 'avg_rating'])).to_numpy(), dtype=tf.float32)
    # Standardizacija user i movie numeričkih
    X_user = X_user_id[:, 1:]
    user_mean = tf.reduce_mean(X_user, axis=0)
    user_std = tf.math.reduce_std(X_user, axis=0)
    X_user_scaled = (X_user - user_mean) / (user_std+ 1e-8)
    X_user_id_scaled = tf.concat([X_user_id[:, :1], X_user_scaled], axis=1)  # Skalirano sa ID kolonom
    movie_mean = tf.reduce_mean(movie_num, axis=0)
    movie_std = tf.math.reduce_std(movie_num, axis=0)
    movie_num_scaled = (movie_num - movie_mean) / (movie_std)
    X_movie_scaled = tf.concat([movie_num_scaled, movie_cat], axis=1)
    # Target skaliranje na [-1, 1]
    y_scaled = 2 * (y - tf.reduce_min(y)) / (tf.reduce_max(y) - tf.reduce_min(y)) - 1
    # scalers = {"user_mean": user_mean, "user_std": user_std,"movie_mean": movie_mean,"movie_std": movie_std, "y_min": tf.reduce_min(y), "y_max": tf.reduce_max(y)}
    if user_id is not None:
        ###Ako je dat user id, filtriramo X_user_id_scaled i X_movie_scaled i vracamo samo korisnika sa tim user_id-om, ako nije vracamo sve korisnike
        maska = tf.reduce_any(tf.equal(tf.expand_dims(X_user_id_scaled[:, 0], 1), tf.constant(user_id, dtype=X_user_id_scaled.dtype)), axis=1)
        X_user_id_scaled = tf.boolean_mask(X_user_id_scaled, maska)  #prva kolona je userid
        y_scaled = tf.boolean_mask(y_scaled, maska)
        return X_user_id_scaled,X_movie_scaled maska , y_scaled#, scalers
    # Ako user_id nije naveden, vracamo sve korisnike bez filtriranja user_id-a
    else:
        return X_user_scaled, X_movie_scaled, y_scaled#, scalers
    
   
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
        user, movies_feat, df = prep_pipeline(batch, movies, batch)
        X_user, X_movie, y = scale(df, user, movies_feat)
        yield (X_user, X_movie), tf.squeeze(y)
        offset += batch_size
    conn.close()

def sql_data_storage(data, kolone):
    engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
    conn = engine.connect()

    first = True
    for (X_user, X_movie), y in tqdm(data):
        user_np = X_user.numpy()
        movie_np = X_movie.numpy()
        y_np = y.numpy().reshape(-1, 1)


        df = pl.DataFrame( data=np.hstack([user_np, movie_np, y_np]),schema=kolone)
        df.write_database('data_storage.ratings', conn, if_table_exists='replace' if first else 'append')
        first = False

    conn.close()

def traning_scalers(kolone):
    engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
    conn = engine.connect()
    q = [f'AVG({i}) as mean_{i}, STDDEV({i}) as sd_{i}' for i in kolone]

    query = f'SELECT {", ".join(q)} FROM data_storage.ratings;'
    
    stats = pl.read_database(query, connection = conn)  
    return stats


###samo RZS?
def prep_tf(user, movies, training_batch = 16):
  user, movies_feat, df = prep_pipeline(ratings, movies)
  X_user, X_movie, y, scalers = scale(df, user, movies_feat)
  data = (X_user, X_movie), y
  data = tf.data.Dataset.from_tensor_slices(data).batch(training_batch)
  return data

def split(data, test_batches = 1):
    test_data = list(data.take(test_batches))
    dev_data = list(data.skip(test_batches).take(test_batches))
    train_data = data.skip(2 * test_batches).prefetch(tf.data.AUTOTUNE).repeat()

    X_user_test = tf.concat([b[0][0] for b in test_data], axis=0)
    X_movie_test = tf.concat([b[0][1] for b in test_data], axis=0)
    y_test = tf.concat([b[1] for b in test_data], axis=0)

    X_user_dev = tf.concat([b[0][0] for b in dev_data], axis=0)
    X_movie_dev = tf.concat([b[0][1] for b in dev_data], axis=0)
    y_dev = tf.concat([b[1] for b in dev_data], axis=0)

    return ((X_user_test, X_movie_test, y_test),
            (X_user_dev, X_movie_dev, y_dev),
            train_data)
    
# def train_test_split(X_user, X_movie, y, test_size=0.2, random_state= 42):       
#     N = X_user.shape[0]
#     tf.random.set_seed(random_state)
#     idx = tf.random.shuffle(tf.range(N))
#     split = int(N * (1 - test_size))
#     train_idx = idx[:split]
#     dev_idx = idx[split:]

#     X_user_train, X_movie_train, y_train = tf.gather(X_user, train_idx), tf.gather(X_movie, train_idx), tf.gather(y, train_idx)

#     X_user_dev, X_movie_dev, y_dev = tf.gather(X_user, dev_idx), tf.gather(X_movie, dev_idx), tf.gather(y, dev_idx)
    
#     return (X_user_train, X_movie_train), y_train, (X_user_dev, X_movie_dev), y_dev




# def NN_prep(df, user, movies, user_id = None):
#     '''
#     Prebacivanje u tenzore i skaliranje --> tf.Tensor
#     user_id - za listu usera, ako je None onda vraca tf.Tensor sa svim userima
#     '''
#     y = tf.convert_to_tensor(df.select(pl.col('rating')).to_series().to_list(), dtype=tf.float32)
#     prva_user = df.columns.index('no genres listed')
#     poslednja_user = df.columns.index('Western')
#     if user_id is None:
#         X_user = tf.convert_to_tensor(df.select(df.columns[prva_user : poslednja_user + 1])
# .to_numpy(), dtype=tf.float32)
#     else:
#         X_user = tf.convert_to_tensor(df.filter(pl.col('userid') == user_id).select(df.columns[prva_user : poslednja_user + 1]).to_numpy(), dtype=tf.float32)
#     X_movie_df = df.select(['year','avg_rating', '#ratings_film'] + [col for col in df.columns if col.endswith('_right')])
#     movie_num = tf.convert_to_tensor(X_movie_df.select(['#ratings_film', 'year', 'avg_rating']).to_numpy(), dtype=tf.float32)
#     movie_cat = tf.convert_to_tensor(X_movie_df.select(pl.all().exclude(['#ratings_film', 'year', 'avg_rating'])).to_numpy(), dtype=tf.float32)
#     # Skaliranje (standardizacija) user i movie numeričkih karakteristika
#     user_mean = tf.reduce_mean(X_user, axis=0)
#     user_std = tf.math.reduce_std(X_user, axis=0)
#     X_user_scaled = (X_user - user_mean) / (user_std + 1e-7)
#     movie_mean = tf.reduce_mean(movie_num, axis=0)
#     movie_std = tf.math.reduce_std(movie_num, axis=0)
#     movie_num_scaled = (movie_num - movie_mean) / (movie_std)
#     X_movie_scaled = tf.concat([movie_num_scaled, movie_cat], axis=1)
#     # Target skaliranje na [-1, 1]
#     y_min = tf.reduce_min(y)
#     y_max = tf.reduce_max(y)
#     y_scaled = 2 * (y - y_min) / (y_max - y_min) - 1

#     # Vrati i transformatore za kasniju upotrebu
#     scalers = {"user_mean": user_mean, "user_std": user_std, "movie_mean": movie_mean, "movie_std": movie_std, "y_min": y_min, "y_max": y_max}

#     return X_user_scaled, X_movie_scaled, y_scaled, scalers

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



