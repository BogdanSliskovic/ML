
from prep import *
from model import ColaborativeFiltering

import polars as pl
from sqlalchemy import create_engine
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import layers, Input, regularizers, Model, optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import joblib
ratings, movies = read_data_lake()
user, movies, df = prep_pipeline(ratings, movies, ratings)

X_user, X_movie, y, scalers = NN_prep(df, user, movies)
print(type(X_user), scalers.keys())

N = X_user.shape[0]
tf.random.set_seed(42)
idx = tf.random.shuffle(tf.range(N))
split = int(N * 0.8)
train_idx = idx[:split]
dev_idx = idx[split:]

X_user_train, X_movie_train, y_train = tf.gather(X_user, train_idx), tf.gather(X_movie, train_idx), tf.gather(y, train_idx)

X_user_dev, X_movie_dev, y_dev = tf.gather(X_user, dev_idx), tf.gather(X_movie, dev_idx), tf.gather(y, dev_idx)


model = ColaborativeFiltering(X_user.shape[1], X_movie.shape[1], embedding=64, learning_rate=0.001)
model.summary()
callbacks = [EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
             ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6, verbose=1)]
history = model.fit(X_user_train, X_movie_train, y_train, X_user_dev, X_movie_dev, y_dev, epochs=20, batch_size=258, callbacks=callbacks)

model.save('model.keras')
joblib.dump(history, 'history.pkl')
joblib.dump(scalers, 'scalers.pkl')

