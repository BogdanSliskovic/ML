import polars as pl
import os
from sqlalchemy import create_engine
import tensorflow as tf
from tqdm import tqdm 

csv_ratings = "../ml-32m/ratings.csv"
csv_movies = "../ml-32m/movies.csv"


def prep_pipeline(ratings, movies):
    '''
    Priprema za model
    '''
    #PROSECAN BROJ OCENA PO FILMU
    num_ratings = ratings.group_by('movieId').agg(pl.len().alias('#ratings_film'))
    user = ratings.join(num_ratings, on = 'movieId', how = 'inner').sort(['movieId', 'userId'])
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
    user_zanr_train = user.join(movies, on='movieId', how='inner')

    #PIVOT LONGER --> ZANROVE PREBACUJEM U JEDNU KOLONU
    user_longer = (user_zanr_train.unpivot(index=['userId', 'rating'],
                                            on=unique_genres).filter(pl.col('value') == 1).rename({'variable': 'genre', 'value': 'is_genre'}))

    #RACUNAM PROSEK ZA SVAKOG USERA ZA SVAKI ZANR I VRACAM U WIDE FORMAT
    user_feature = user_longer.group_by('userId').agg([(pl.when(pl.col('genre') == genre).then(pl.col('rating')).mean().alias(genre)) for genre in unique_genres]).fill_null(0)
    movie_avg_rating = (user.group_by('movieId').agg(pl.col('rating').mean().alias('avg_rating')))
    movie_features = movies.join(movie_avg_rating, on='movieId', how='left').fill_null(0)
    movie_features = movie_features.select(['movieId', 'title','year','avg_rating', *unique_genres])

    X_user = ratings.join(user_feature, on="userId", how="inner").drop('rating').collect()

    X_movie = user.join(movie_features, on="movieId", how="inner").drop('rating', 'title').collect()
    
    return X_user, X_movie

def get_genres(movies, prep = False):
    movies = movies.with_columns(pl.col("genres").str.split("|"))
    unique_genres = sorted(set(g for genre in movies["genres"] for g in genre))
    unique_genres[0] = unique_genres[0].replace('(', '').replace(')', '')
    if prep == True:
      return movies, unique_genres
    else:
      return unique_genres
  

def batch_generator(batch_size=1_000_000):
    """
    Pravi batch-eve iz CSV fajla sa ocenjivanjem filmova (ratings).
    """
    movies = pl.read_csv(csv_movies)
    offset = 0
    #Lazy prebroji redove
    total_rows = pl.scan_csv(csv_ratings).select(pl.len()).collect()[0, 0]

    while offset < total_rows:
        batch = pl.read_csv(csv_ratings).slice(offset, batch_size)
        if batch.height == 0:
            break

        user, movie = prep_pipeline(batch, movies)
        
        user = tf.convert_to_tensor(user.to_numpy(), dtype=tf.float64)
        movie = tf.convert_to_tensor(movie.to_numpy(), dtype=tf.float64)
        yield (user, movie)

        offset += batch_size

data = tf.data.Dataset.from_generator(lambda: batch_generator(), output_signature= (
    tf.TensorSpec(shape=(None, 22), dtype=tf.float64, name = 'user'),  # User features
    tf.TensorSpec(shape=(None, 25), dtype=tf.float64, name = 'movie')   # Movie features
))


from tensorflow.keras.layers import Normalization

# Kreiramo normalizacione layere
norm_user = Normalization(axis = 1)
norm_movie = Normalization()

# OVO GA "UČI" (računa mean i variance)
norm_user.adapt(user_ds)
norm_movie.adapt(movie_ds)

tf.math.sqrt(norm_user.variance)

norm_user(user[:, 2:])
for user, movie in data.take(1):
    user_normed = norm_user(user[:, 2:])

    print("Standardizovan user:", user_normed)

print("Means po kolonama:", norm_user.mean.numpy())
print("Variance po kolonama:", norm_user.variance.numpy())

for user, movie in data.take(2):
    print("User std:", tf.math.reduce_std(user[:, 2:], axis = 0).numpy())
    print("Movie std:", tf.math.reduce_std(movie[:, 2:], axis = 0).numpy())

stats = get_mean(data)
for statisika in stats.keys():
    for df in stats[statisika].keys():
        globals()[f'{statisika}_{df}'] = stats[statisika][df]
        
        
std_movie
std_user
        
        print(stats[statisika][df])
stats['mean']['user']
stats['mean']['movie']
tf.print(stats['mean']['user'], summarize = 20)

ratings.group_by('userId').mean()['rating'].mean()


from model import ColaborativeFiltering
from prep import *
import tensorflow as tf
from tqdm import tqdm 

csv_ratings = "../ml-32m/ratings.csv"
csv_movies = "../ml-32m/movies.csv"

data = tf.data.Dataset.from_generator(lambda: batch_generator(batch_size= 1_000_000), output_signature= (tf.TensorSpec(shape=(None, 22), dtype=tf.float64, name = 'user'), tf.TensorSpec(shape=(None, 25), dtype=tf.float64, name = 'movie')))







data = tf.data.Dataset.from_generator(lambda: batch_generator(ratings_path= csv_ratings, movies_path= csv_movies,batch_size= 1_000), output_signature= (tf.TensorSpec(shape=(None, 22), dtype=tf.float64, name = 'user'), tf.TensorSpec(shape=(None, 25), dtype=tf.float64, name = 'movie')))

next(iter(data))

model = ColaborativeFiltering(num_user_features=22, num_movie_features= 25)

ratings = pl.read_csv(csv_ratings)[:1000]
movies = pl.read_csv(csv_movies)
user, movies = prep_pipeline(ratings, movies)


engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
conn = engine.connect()
pl.read_database('select * from data_storage.movies', connection=conn)

def to_data_storage():
    engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
    conn = engine.connect()
    prvi = True

for user, movies in data.take(1):
    pl.DataFrame(user.numpy()).write_database('data_storage.user', conn, if_table_exists='replace' if prvi else 'append')
    pl.DataFrame(movies.numpy()).write_database('data_storage.movies', conn, if_table_exists='replace' if prvi else 'append')
    
pl.read_database('select * from data_storage.movies', connection=conn)



def tf_dataset_to_sql(dataset, conn):
    prvi = True
    i = 0
    for user_tensor, movies_tensor in dataset:
        # konvertuj tensor u polars dataframe
        df_user = pl.DataFrame(user_tensor.numpy())
        df_movies = pl.DataFrame(movies_tensor.numpy())
        i+=1
        print(i)
        # upis u bazu
        
        df_user.write_database('data_storage.user', conn, if_table_exists='replace' if prvi else 'append')
        df_movies.write_database('data_storage.movies', conn, if_table_exists='replace' if prvi else 'append')
        prvi = False
        if i ==10:
            break
engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
conn = engine.connect()

# Pozovi funkciju
tf_dataset_to_sql(data, conn)

conn.close()

def batch_generator(ratings_path, movies_path, batch_size=1_500_000):
    """
    Pravi batch-eve iz CSV fajla sa ocenjivanjem filmova (ratings).
    """
    movies = pl.read_csv(movies_path)
    offset = 0
    #Lazy prebroji redove
    total_rows = pl.scan_csv(ratings_path).select(pl.len()).collect()[0, 0]
    while offset < total_rows:
        batch = pl.read_csv(ratings_path).slice(offset, batch_size)
        if batch.height == 0:
            break
        user_df, movies_df= prep_pipeline(batch, movies)
        user_tensor = tf.convert_to_tensor(user_df.to_numpy(), dtype=tf.float64)
        movies_tensor = tf.convert_to_tensor(movies_df.to_numpy(), dtype=tf.float64)
        
        yield (user_tensor, movies_tensor)
        
        offset += batch_size

data = tf.data.Dataset.from_generator(lambda: batch_generator(ratings_path= csv_ratings, movies_path= csv_movies,batch_size= 1_000), output_signature= (tf.TensorSpec(shape=(None, 22), dtype=tf.float64, name = 'user'), tf.TensorSpec(shape=(None, 25), dtype=tf.float64, name = 'movie')))

for i in data.take(3):
    print(i)


def data_storage(data):
    engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
    prvi = True
    i = 0
    for u, m in tqdm(data):
        conn = engine.connect()
        pl.DataFrame(u.numpy()).write_database('data_storage.user', conn, if_table_exists='replace' if prvi else 'append')
        pl.DataFrame(m.numpy()).write_database('data_storage.movies', conn, if_table_exists='replace' if prvi else 'append')
        prvi = False
        i+=1
        print(i)
data_storage(data)
    
pl.read_database('SELECT * FROM data_storage.user', connection=conn)

