from prep import *
from model import ColaborativeFiltering
from sqlalchemy import create_engine
import polars as pl
import os
import tensorflow as tf
from keras import layers, Input, regularizers, Model, optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
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

user, movies_feat, df = prep_pipeline(ratings, movies)
X_user, X_movie, y, scalers = scale(df, user, movies_feat)

def prep_tf(user, movies, training_batch = 16):
  user, movies_feat, df = prep_pipeline(ratings, movies)
  X_user, X_movie, y, scalers = scale(df, user, movies_feat)
  data = (X_user, X_movie), y
  data = tf.data.Dataset.from_tensor_slices(data).batch(training_batch)
  return data
###
def split(data):
  (X_user_test, X_movie_test), y_test = next(iter(data))
  (X_user_dev, X_movie_dev), y_dev = next(iter(data.skip(1)))
  train_data = data.skip(2).prefetch(tf.data.AUTOTUNE).repeat()
  return ((X_user_test, X_movie_test, y_test), (X_user_dev, X_movie_dev, y_dev), train_data)

data = prep_tf(ratings, movies)
test_set, dev_set, train_data = split(data)

X_user_test, X_movie_test, y_test = test_set
X_user_dev, X_movie_dev, y_dev = dev_set


model = ColaborativeFiltering(20, 23 ,user_layers = [256, 128, 64],embedding=64, learning_rate=0.001)#, user_reg = [regularizers.l2(0.01), None, None])
callbacks = [EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)]
history = model.fit(train_data, epochs=50,validation_data = ((X_user_dev, X_movie_dev), y_dev), callbacks=callbacks, steps_per_epoch = int(10000/16))


///



engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
conn = engine.connect()
movies = pl.read_database(query='SELECT * FROM raw.movies', connection=conn)
conn.close()

total = 50000

data = tf.data.Dataset.from_generator(
    lambda: batch_generator(movies, batch_size=4096, total = total),
    output_signature=(
        (tf.TensorSpec(shape=(None, 20), dtype=tf.float32, name= 'X_user'),
         tf.TensorSpec(shape=(None, 23), dtype=tf.float32, name='X_movie')),
        tf.TensorSpec(shape=(None,), dtype=tf.float32, name='y')
    )
)

(X_user_dev, X_movie_dev), y_dev = next(iter(data))
training_batch = 4096
train_data = data.unbatch().batch(training_batch).skip(1).prefetch(tf.data.AUTOTUNE).repeat()

# import numpy as np
# np.isnan(X_user_dev.numpy()).any()
# np.isnan(X_movie_dev.numpy()).any()
# np.isnan(y_dev.numpy()).any()



# for batch in train_data.take(5):
#     (X_user_batch, X_movie_batch), y_batch = batch
#     print("X_user_batch shape:", X_user_batch.shape)
#     print("X_movie_batch shape:", X_movie_batch.shape)
#     print("y_batch shape:", y_batch.shape)
#     print("X_user_batch:", X_user_batch.numpy()[0])
#     print("X_movie_batch:", X_movie_batch.numpy()[0])
#     print("y_batch:", y_batch.numpy()[0])

for batch in train_data.take(15):
    (X_user_batch, X_movie_batch), y_batch = batch
    print(np.isnan(X_user_batch.numpy()).any(), np.isnan(X_movie_batch.numpy()).any(), np.isnan(y_batch.numpy()).any())
    print(np.isinf(X_user_batch.numpy()).any(), np.isinf(X_movie_batch.numpy()).any(), np.isinf(y_batch.numpy()).any())
    print("y_batch min/max:", y_batch.numpy().min(), y_batch.numpy().max())

# for i, batch in enumerate(train_data.take(8)):
#     (X_user_batch, X_movie_batch), y_batch = batch
#     nan_mask = np.isnan(X_user_batch.numpy()).any(axis=1)
#     if nan_mask.any():
#         print(f"Batch {i+1} ima NaN u X_user_batch na indeksima:", np.where(nan_mask)[0])
#         print("Redovi sa NaN:", X_user_batch.numpy()[nan_mask])
#         # Opcionalno: pogledaj i y_batch[nan_mask], X_movie_batch[nan_mask]



model = ColaborativeFiltering(20, 23 ,user_layers = [256, 128, 64],embedding=64, learning_rate=0.001)#, user_reg = [regularizers.l2(0.01), None, None])
model.summary()
callbacks = [EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)]
history = model.fit(train_data, epochs=20, validation_data=([X_user_dev,  X_movie_dev], y_dev), callbacks=callbacks, steps_per_epoch = int(total // training_batch))


model.save('model_proba.keras')
joblib.dump(history, 'history_proba.pkl')
joblib.dump(scalers, 'scalers_proba.pkl')

# from tensorflow.keras.models import load_model
# load_model('model_proba.keras')



