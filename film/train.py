from prep import *
from model import *
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.saving import register_keras_serializable
import joblib
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import layers
from keras.saving import register_keras_serializable
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import layers
from keras.saving import register_keras_serializable

csv_ratings = "../ml-32m/ratings.csv"
csv_movies = "../ml-32m/movies.csv"

user_kolone, movies_kolone = imena_kolona(csv_ratings, csv_movies)

batch = 100 #~65536


data = tf.data.Dataset.from_generator(lambda: batch_generator(ratings_path= csv_ratings, movies_path= csv_movies,batch_size= batch, train = True), output_signature= ((tf.TensorSpec(shape=(None, 20), dtype=tf.float64, name = 'user'), tf.TensorSpec(shape=(None, 23), dtype=tf.float64, name = 'movie')), tf.TensorSpec(shape=(None,1), dtype=tf.float32)))


train_size = 2#(10_000_000 // batch) +1 #10_027_008
dev_size = 1 #(250_000 // batch) + 1 #262144
#test_size = #(250_000 // batch) + 1 #262144

data_train = data.take(train_size).prefetch(tf.data.AUTOTUNE) #ne stavlja se repeat zbog adapt (beskonacno ga racuna posto nema kraja datasetu)


norm_user = standardizacija(data_train.map(lambda x,y: x[0]))
norm_movies = standardizacija(data_train.map(lambda x,y: x[1][:,:3])) # samo prve tri kolone su numericke (#ratings_film, year, avg rating, ostale su dummy)
keras.models.save_model(norm_user, '123.keras')
t = keras.models.load_model('123.keras')
t.mean
norm_user

data_train = data_train.map(lambda x, y: ((norm_user(x[0]),tf.concat([tf.cast(norm_movies(x[1][:, :3]), tf.float32),tf.cast(x[1][:, 3:], tf.float32)], axis=1)),scale_y(y))).repeat().prefetch(tf.data.AUTOTUNE)

model = MovieRecommender(user_input_dim=20,movie_input_dim=23,hidden_layers=[128, 128, 64], embedding_dim=32)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae', 'mse'])

data_dev = data.skip(train_size).take(dev_size)

data_dev = data_dev.map(lambda x, y: ((norm_user(x[0]),tf.concat([tf.cast(norm_movies(x[1][:, :3]), tf.float32),tf.cast(x[1][:, 3:], tf.float32)], axis=1)),scale_y(y))).prefetch(tf.data.AUTOTUNE)


history = model.fit(data_train, validation_data=data_dev, epochs=5, steps_per_epoch=train_size, validation_steps=dev_size)


# model.save(f'model_128_128_64_32.keras')
keras.models.save_model(norm_user, 'norm_user.keras')
keras.models.save_model(norm_movies, 'norm_movies.keras')
# joblib.dump(history, 'histori_128_128_64_32.pkl')

def serialize_example(user, movie, label):
    feature = {
        'user': tf.train.Feature(float_list=tf.train.FloatList(value=user)),
        'movie': tf.train.Feature(float_list=tf.train.FloatList(value=movie)),
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=label)),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

with tf.io.TFRecordWriter('dataset.tfrecord') as writer:
    for (x, y) in data_train.take(15):
        user, movie = x
        label = y
        for i in range(user.shape[0]):
            serialized = serialize_example(user[i].numpy(), movie[i].numpy(), label[i].numpy())
            writer.write(serialized)
