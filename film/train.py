from prep import *
import polars as pl
from sqlalchemy import create_engine
import tensorflow as tf
from sklearn.model_selection import train_test_split

ratings, movies = read_data_lake()
user, movies, df = prep_pipeline(ratings, movies, ratings)

X_user, X_movie, y = NN_prep(df, user, movies)
print(type(X_user))

N = X_user.shape[0]
tf.random.set_seed(42)
idx = tf.random.shuffle(tf.range(N))
split = int(N * 0.8)
train_idx = idx[:split]
dev_idx = idx[split:]

X_user_train = tf.gather(X_user, train_idx)
X_movie_train = tf.gather(X_movie, train_idx)
y_train = tf.gather(y, train_idx)

X_user_dev = tf.gather(X_user, dev_idx)
X_movie_dev = tf.gather(X_movie, dev_idx)
y_dev = tf.gather(y, dev_idx)