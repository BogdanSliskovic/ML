import polars as pl
from sqlalchemy import create_engine
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
np.set_printoptions(suppress=True)
import joblib


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time

engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
conn = engine.connect()

ratings = pl.read_database(query='SELECT * FROM raw.ratings ORDER BY RANDOM() LIMIT 2500000', connection=conn)
movies = pl.read_database(query='SELECT * FROM raw.movies', connection=conn)
conn.close()

conn = engine.connect()
ratings.write_database(
    table_name="data_lake.ratings",
    connection=conn,
    if_table_exists="replace",
)

print("Uspesno upisani redovi")
conn.close()

# start_time = time.time()
# engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
# conn = engine.connect()
# ratings = pl.read_database(query='SELECT * FROM data_lake.ratings', connection=conn)
# movies = pl.read_database(query='SELECT * FROM raw.movies', connection=conn)
# print(f"Execution time: {time.time() - start_time:.4f} seconds")
# ratings.describe()[['statistic', 'rating']]
# movies

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
    # #U NUMPY I SKALIRANJE
    
    y = df.select(pl.col('rating')).to_numpy()
    X_user = df.select(user.select(pl.exclude('userid')).columns)
    X_movie = df.select(movies.select(pl.exclude(['movieid','title'])).columns + ['#ratings_film'])
    SS_movie = StandardScaler()
    SS_user = StandardScaler()
    SS_target = MinMaxScaler((-1,1))
    movie_num = SS_movie.fit_transform(X_movie['#ratings_film', 'year', 'avg_rating'])
    movie_cat = X_movie.select(pl.all().exclude(['#ratings_film', 'year', 'avg_rating'])).to_numpy()
    X_movie_numpy = np.column_stack([movie_num,movie_cat])
    X_user_numpy = SS_user.fit_transform(X_user)
    y = SS_target.fit_transform(y)
    return X_user_numpy, X_movie_numpy, y, SS_movie, SS_user, SS_target

X_user_numpy, X_movie_numpy, y, SS_movie, SS_user, SS_target = NN_prep(df)

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

X_user_train, X_user_dev, X_movie_train, X_movie_dev, y_train, y_dev = train_test_split(X_user_numpy, X_movie_numpy,
                                                                                        y, test_size=0.15, random_state=42)
X_user_train.shape, X_user_dev.shape, X_movie_train.shape, X_movie_dev.shape, y_train.shape, y_dev.shape
X_user_train[:5], X_movie_train[:5], y_train[:5]

num_user_features = X_user_numpy.shape[1]
num_item_features = X_movie_numpy.shape[1]
num_outputs = 64

# ---------------- USER MODEL ----------------
user_input = keras.Input(shape=(num_user_features,), name='user_input')
x_user = layers.Dense(128, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer= keras.regularizers.l2(0.001))(user_input)
x_user = layers.Dense(num_outputs, activation='tanh', kernel_initializer='glorot_uniform')(x_user)
user_embedding = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(x_user)

# ---------------- ITEM MODEL ----------------
item_input = keras.Input(shape=(num_item_features,), name='item_input')
x_item = layers.Dense(128, activation='tanh', kernel_initializer='glorot_uniform')(item_input)
x_item = layers.Dense(num_outputs, activation='tanh', kernel_initializer='glorot_uniform')(x_item)
item_embedding = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(x_item)

# ---------------- COSINE SIMILARITY ----------------
cos_sim = layers.Dot(axes=1, name='cosine_similarity')([user_embedding, item_embedding])
# rezultat je u [-1, 1]

# ---------------- FINAL MODEL ----------------
model_tanh_l2 = keras.Model(inputs=[user_input, item_input], outputs=cos_sim)

# Kompajliraj model
model_tanh_l2.compile(optimizer= keras.optimizers.Nadam(learning_rate= 0.001), loss='mse', metrics=['mae', 'mse'])

# Prikaži arhitekturu
model_tanh_l2.summary()
##12.38
early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True,)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6, verbose=1)

history_tanh_l2 = model_tanh_l2.fit(x=[X_user_train, X_movie_train],y=y_train, callbacks=[early_stop, reduce_lr],
    validation_data=([X_user_dev, X_movie_dev], y_dev),epochs=25, batch_size=512)


help(model_tanh_l2.fit)

history_tanh_l2.history.keys()
import matplotlib.pyplot as plt


model_tanh_l2.save('model_tanh_l2.keras')
joblib.dump(SS_movie, 'SS_movie.pkl')
joblib.dump(SS_user, 'SS_user.pkl') 
joblib.dump(SS_target, 'SS_target.pkl')
joblib.dump(history_tanh_l2, 'history_tanh_l2.pkl')

#model_tanh_l2 = keras.models.load_model('model_tanh_l2.h5', custom_objects={'cosine_similarity': tf.keras.backend.function})
#history_tanh_l2 = joblib.load('history_tanh_l2.pkl')
plt.plot(history_tanh_l2.history['loss'], label='loss')
plt.plot(history_tanh_l2.history['val_loss'], label='val_loss')
