from prep import *

import tensorflow as tf
from tensorflow import keras
import joblib
import matplotlib.pyplot as plt
ratings, movies = read_data_lake()
user, movies, df = prep_pipeline(ratings, movies, ratings)

X_user, X_movie, y, scalers = NN_prep(df, user, movies)

model = keras.models.load_model('model.keras', safe_mode=False)
history = joblib.load('history.pkl')
import pandas as pd
pd.Series(history.history['loss']).plot()