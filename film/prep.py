import polars as pl
import tensorflow as tf

def prep_pipeline(ratings, movies):
    '''
    Priprema za model
    '''
    #PROSECAN BROJ OCENA PO FILMU
    num_ratings = ratings.group_by('movieId').agg(pl.len().alias('#ratings_film'))
    user = ratings.join(num_ratings, on = 'movieId', how = 'left')
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
    user_zanr_train = user.join(movies, on='movieId', how='left')

    #PIVOT LONGER --> ZANROVE PREBACUJEM U JEDNU KOLONU
    user_longer = (user_zanr_train.unpivot(index=['userId', 'rating'],
                                            on=unique_genres).filter(pl.col('value') == 1).rename({'variable': 'genre', 'value': 'is_genre'}))

    #RACUNAM PROSEK ZA SVAKOG USERA ZA SVAKI ZANR I VRACAM U WIDE FORMAT
    user_feature = user_longer.group_by('userId').agg([(pl.when(pl.col('genre') == genre).then(pl.col('rating')).mean().alias(genre)) for genre in unique_genres]).fill_null(0)
    movie_avg_rating = (user.group_by('movieId').agg(pl.col('rating').mean().alias('avg_rating')))
    movie_features = movies.join(movie_avg_rating, on='movieId', how='left').fill_null(0)
    movie_features = movie_features.select(['movieId', 'title','year','avg_rating', *unique_genres])

    X_user = ratings.join(user_feature, on="userId", how = 'left').fill_null(0).drop('rating').collect()

    X_movie = user.join(movie_features, on="movieId", how="left").drop('rating', 'title').collect()
    
    y = ratings.collect()
    
    return X_user, X_movie, y
# ratings.collect()

def get_genres(movies, prep = False):
    movies = movies.with_columns(pl.col("genres").str.split("|"))
    unique_genres = sorted(set(g for genre in movies["genres"] for g in genre))
    unique_genres[0] = unique_genres[0].replace('(', '').replace(')', '')
    if prep == True:
      return movies, unique_genres
    else:
      return unique_genres

def batch_generator(ratings_path, movies_path, batch_size=1_500_000, train = False):
    """
    Pravi batch-eve iz CSV fajla sa ocenjivanjem filmova (ratings).
    """
    movies = pl.read_csv(movies_path)
    offset = 0
    #Lazy prebroji redove
    total_rows = pl.scan_csv(ratings_path).select(pl.len()).collect()[0, 0]
    while offset < total_rows:
        batch = pl.read_csv(ratings_path).slice(offset, batch_size)
        if batch.height < batch_size:
            break
        user_df, movies_df, y_df= prep_pipeline(batch, movies)

        user_tensor = tf.convert_to_tensor(user_df.to_numpy(), dtype=tf.float32)
        movies_tensor = tf.convert_to_tensor(movies_df.to_numpy(), dtype=tf.float32)
        y = tf.convert_to_tensor(y_df.to_numpy(), dtype=tf.float32)
        if train: #ako je za trening vrati bez ID
            yield (user_tensor[:,2:], movies_tensor[:,2:]),  y[:, 2:]
        else: #ako je za inference prve dve kolone su ID u svakom tenzoru
            yield (user_tensor, movies_tensor), y
        
        offset += batch_size
        
def imena_kolona(ratings_path, movies_path):
    ratings = pl.read_csv(ratings_path).slice(0,10)
    movies = pl.read_csv(movies_path)
    user, movies, _ = prep_pipeline(ratings, movies)
    return user.columns, movies.columns
