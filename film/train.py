from prep import *
from model import *
import tensorflow as tf

from tensorflow import keras

import joblib
import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
from tensorflow import keras
import numpy as np


batch = 2**16 #65536

csv_ratings = "../ml-32m/ratings.csv"
csv_movies = "../ml-32m/movies.csv"


#Podela na train, dev i test u razlicite .csv fajlove
za_test = (1_000_000 // batch + 1) * batch

df = pl.read_csv(csv_ratings)
#promesaj df
dev = pl.read_csv(csv_ratings).sample(za_test, shuffle= True, seed = 42)
test = pl.read_csv(csv_ratings).sample(za_test, shuffle= True, seed = 42)

dev_test = pl.concat([dev, test])

train = df.join(dev_test, on = ['userId', 'movieId'], how = 'anti')
train = train.sample(fraction=1.0, shuffle=True, seed = 42) #promesaj train set

for skup in ['train', 'dev', 'test']:
    temp = globals()[skup]
    path = f'../ml-32m/ratings_{skup}.csv'
    temp.write_csv(path)
    globals()[skup+ '_path'] = path



data_train = tf.data.Dataset.from_generator(lambda: batch_generator(ratings_path= train_path, movies_path= csv_movies,batch_size= batch, train = True),
    output_signature= ((tf.TensorSpec(shape=(None, 20), dtype=tf.float64, name = 'user'), tf.TensorSpec(shape=(None, 23),
    dtype=tf.float64, name = 'movie')), tf.TensorSpec(shape=(None,1), dtype=tf.float32))).prefetch(tf.data.AUTOTUNE)

data_dev = tf.data.Dataset.from_generator(lambda: batch_generator(ratings_path= dev_path, movies_path= csv_movies,batch_size= batch, train = True),
    output_signature= ((tf.TensorSpec(shape=(None, 20), dtype=tf.float64, name = 'user'), tf.TensorSpec(shape=(None, 23),
    dtype=tf.float64, name = 'movie')), tf.TensorSpec(shape=(None,1), dtype=tf.float32))).prefetch(tf.data.AUTOTUNE)

user_kolone, movies_kolone = imena_kolona(csv_ratings, csv_movies)

# norm_user = standardizacija(data_train.map(lambda x,y: x[0]))
# norm_movies = standardizacija(data_train.map(lambda x,y: x[1][:,:3])) # samo prve tri kolone su numericke (#ratings_film, year, avg rating, ostale su dummy)
# keras.models.save_model(norm_user, 'user_scaler.keras')
# keras.models.save_model(norm_movies, 'movies_scaler.keras')
norm_user = keras.models.load_model('user_scaler.keras')
norm_movies = keras.models.load_model('movies_scaler.keras')

#skaliranje
data_train = data_train.map(lambda x, y: ((norm_user(x[0]),tf.concat([tf.cast(norm_movies(x[1][:, :3]), tf.float32),tf.cast(x[1][:, 3:], tf.float32)], axis=1)),scale_y(y))).repeat().prefetch(tf.data.AUTOTUNE)
next(iter(data_train))
data_dev = data_dev.map(lambda x, y: ((norm_user(x[0]),tf.concat([tf.cast(norm_movies(x[1][:, :3]), tf.float32),tf.cast(x[1][:, 3:], tf.float32)], axis=1)),scale_y(y))).prefetch(tf.data.AUTOTUNE)

model = MovieRecommender(user_input_dim=20,movie_input_dim=23,hidden_layers=[128, 128, 64], embedding_dim=32)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae', 'mse'])

train_size = train.height // batch

history = model.fit(data_train, validation_data=data_dev, epochs=15, steps_per_epoch=train_size)


model.save(f'model_128_128_64_32.keras')
keras.models.save_model(norm_user, 'user_scaler.keras')
keras.models.save_model(norm_movies, 'movies_scaler.keras')
joblib.dump(history, 'histori_128_128_64_32.pkl')
