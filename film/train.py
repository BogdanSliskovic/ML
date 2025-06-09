from prep import *
from model import ColaborativeFiltering
from sqlalchemy import create_engine
import polars as pl
import os
import tensorflow as tf
from keras import layers, Input, regularizers, Model, optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import numpy as np
///
###TRAIN.py

###
#samo za RZS
movies = pl.read_csv(r'https://raw.githubusercontent.com/BogdanSliskovic/ML/refs/heads/main/film/movies.csv')
movies.name = 'Movies'
ratings = pl.read_csv(r'https://raw.githubusercontent.com/BogdanSliskovic/ML/refs/heads/main/film/ratings_RZS.csv')
ratings.name = 'Ratings'

for df in [movies, ratings]:
  print(df.name , df.schema, df.shape)

user, movies_feat, df = prep_pipeline(ratings__, movies)
kolone = df.drop('title').columns
kolone
###RZS
# X_user, X_movie, y, scalers = scale(df, user, movies_feat)


# data = prep_tf(ratings, movies)

 
# (X_user_test, X_movie_test, y_test), (X_user_dev, X_movie_dev, y_dev), train_data = split(data)


# model = ColaborativeFiltering(20, 23 ,user_layers = [256, 128, 64],embedding=64, learning_rate=0.001)#, user_reg = [regularizers.l2(0.01), None, None])
# callbacks = [EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2 min_lr=1e-6, verbose=1)]
# history = model.fit(train_data, epochs=5,validation_data = ((X_user_dev, X_movie_dev), y_dev), callbacks=callbacks, steps_per_epoch = int(10000/16))
///
num_ratings = ratings.group_by('movieid').agg(pl.len().alias('#ratings_film'))
user = ratings.join(num_ratings, on = 'movieid', how = 'inner').sort(['movieid', 'userid'])
movies, unique_genres = get_genres(movies, prep = True)
#LAZY!
# user = user.lazy()
# movies = movies.lazy()
# ratings = ratings.lazy()
#SVI ZANROVI
for genre in unique_genres:
    movies = movies.with_columns(pl.col("genres").list.contains(genre).cast(pl.Int8).alias(genre))
movies = movies.drop('genres')
#KOLONA GODINA
movies = movies.with_columns(pl.col("title").str.extract(r"\((\d{4})\)", 1).cast(pl.Int16).alias("year"))

#ISTI FORMAT TABELE KAO MOVIES
user_zanr_train = user.join(movies, on='movieid', how='inner')

#PIVOT LONGER --> ZANROVE PREBACUJEM U JEDNU KOLONU
user_longer = (user_zanr_train.unpivot(index=['userid', 
'rating' ],                                        on=unique_genres).filter(pl.col('value') == 1).rename({'variable': 'genre', 'value': 'is_genre'}))

#RACUNAM PROSEK ZA SVAKOG USERA ZA SVAKI ZANR I VRACAM U WIDE FORMAT
user_feature = user_longer.group_by('userid').agg([(pl.when(pl.col('genre') == genre).then(pl.col('rating')).mean().alias(genre)) for genre in unique_genres]).fill_null(0)
movie_avg_rating = (user.group_by('movieid').agg(pl.col('rating').mean().alias('avg_rating')))
movie_features = movies.join(movie_avg_rating, on='movieid', how='left').fill_null(0)
movie_features = movie_features.select(['movieid', 'title','year','avg_rating', *unique_genres])

X_user = ratings.join(user_feature, on="userid", how="inner").drop('rating').collect()

X_movie = user.join(movie_features, on="movieid", how="inner").drop('rating', 'title').collect()


X_movie.columns

X_user, X_movie = user_feature.collect(), movie_features.collect()

///


prva_user = df.columns.index('no genres listed')
poslednja_user = df.columns.index('Western')
X_user_id = tf.convert_to_tensor(df.select(['userid', 'movieid'] + df.columns[prva_user : poslednja_user + 1]).to_numpy(), dtype=tf.float32)
X_movie_df = df.select(['movieid','userid', 'year','avg_rating', '#ratings_film'] + [col for col in df.columns if col.endswith('_right')])
movie_num = tf.convert_to_tensor(X_movie_df.select(['#ratings_film', 'year', 'avg_rating']).to_numpy(), dtype=tf.float32)
movie_cat = tf.convert_to_tensor(X_movie_df.select(pl.all().exclude(['#ratings_film', 'year', 'avg_rating'])).to_numpy(), dtype=tf.float32)
//
engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
conn = engine.connect()
movies = pl.read_database(query='SELECT * FROM raw.movies', connection=conn)
conn.close()

total = 2000000
training_batch = 100000
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
        # if batch.height == 0:
            # break
        user, movies, _ = prep_pipeline(batch, movies)
        df = df.drop('title')
        
        # X_user, X_movie, y = scale(df, user, movies_feat)
        # yield (X_user, X_movie), tf.squeeze(y)
        df = tf.convert_to_tensor(df.to_numpy(), dtype=tf.float32)
        yield df
        offset += batch_size
    conn.close()

data = tf.data.Dataset.from_generator(lambda: batch_generator(movies, batch_size=training_batch, total=total), output_signature=(tf.TensorSpec(shape=(None, 46), dtype=tf.float32, name='df')))

engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
conn = engine.connect()
proba = pl.read_database(query='SELECT * FROM data_storage.ratings limit 1', connection=conn)
conn.close()

len(kolone)
len(df.columns)
df.columns[20:]
next(iter(train_data))
//
engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
conn = engine.connect()

first = True
for df in tqdm(data):
    temp = pl.DataFrame(df.numpy(), schema = kolone)
    temp.write_database('data_storage.ratings', conn, if_table_exists='replace' if first else 'append')
    first = False

def traning_scalers(kolone):
    engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
    conn = engine.connect()
    q = [f'AVG(\"{i}\") as mean_{i}, STDDEV(\"{i}\") as sd_{i}' for i in kolone]

    query = f'SELECT {", ".join(q)} FROM data_storage.ratings;'
    
    stats = pl.read_database(query, connection = conn)  
    return stats
traning_scalers(kolone)
# temp
# engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
# conn = engine.connect()
# p = pl.read_database(query='SELECT * FROM data_storage.ratings', connection=conn)
# conn.close()


df = next(iter(data))

sql_data_storage(train_data, kolone)
//

model = ColaborativeFiltering(20, 23 ,user_layers = [256, 128, 64],embedding=64, learning_rate=0.001)#, user_reg = [regularizers.l2(0.01), None, None])
model.summary()
callbacks = [EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)]
history = model.fit(train_data, epochs=20, validation_data=([X_user_dev,  X_movie_dev], y_dev), callbacks=callbacks, steps_per_epoch = int(total // training_batch))

movies



model.save('model_proba.keras')
joblib.dump(history, 'history_proba.pkl')
joblib.dump(scalers, 'scalers_proba.pkl')

# from tensorflow.keras.models import load_model
# load_model('model_proba.keras')

# Export batches to SQL using the batch_generator, not the tf.data.Dataset
from tqdm import tqdm
engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
conn = engine.connect()
first = True
for _, _, df_batch in tqdm(batch_generator(movies, batch_size=training_batch, total=total)):
    df_batch.write_database('data_storage.ratings', conn, if_table_exists='replace' if first else 'append')
    first = False
conn.close()



