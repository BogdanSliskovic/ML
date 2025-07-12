from prep import *
from model import *
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.saving import register_keras_serializable
import joblib

csv_ratings = "../ml-32m/ratings.csv"
csv_movies = "../ml-32m/movies.csv"

df = pl.read_csv(csv_ratings)

user_kolone, movies_kolone = imena_kolona(csv_ratings, csv_movies)

batch = 2**16 #~65536
4*batch

data = tf.data.Dataset.from_generator(lambda: batch_generator(ratings_path= csv_ratings, movies_path= csv_movies,batch_size= batch, train = True), output_signature= ((tf.TensorSpec(shape=(None, 20), dtype=tf.float64, name = 'user'), tf.TensorSpec(shape=(None, 23), dtype=tf.float64, name = 'movie')), tf.TensorSpec(shape=(None,1), dtype=tf.float32)))


train_size = (10_000_000 // batch) +1 #10_027_008
dev_size = (250_000 // batch) + 1 #262144
test_size = (250_000 // batch) + 1 #262144

data_train = data.take(train_size).prefetch(tf.data.AUTOTUNE) #ne stavlja se repeat zbog adapt (beskonacno ga racuna posto nema kraja datasetu)


norm_user = standardizacija(data_train.map(lambda x,y: x[0]))
norm_movies = standardizacija(data_train.map(lambda x,y: x[1][:,:3])) # samo prve tri kolone su numericke (#ratings_film, year, avg rating, ostale su dummy)


data_train = data_train.map(lambda x, y: ((norm_user(x[0]),tf.concat([tf.cast(norm_movies(x[1][:, :3]), tf.float32),tf.cast(x[1][:, 3:], tf.float32)], axis=1)),scale_y(y))).repeat().prefetch(tf.data.AUTOTUNE)

# Ulazi
user_input = tf.keras.Input(shape=(20,), name="user_input")
movie_input = tf.keras.Input(shape=(23,), name="movie_input")

slojevi = [128,128, 64]
embedding = 32

u = user_input
for sloj in slojevi:
    u = layers.Dense(sloj, kernel_initializer='he_normal')(u)
    u = layers.BatchNormalization()(u)
    u = layers.Activation('relu')(u)
u = layers.Dense(embedding, activation='relu', kernel_initializer='he_normal')(u)
user_embedding = L2NormalizeLayer()(u)

m = movie_input
for sloj in slojevi:
    m = layers.Dense(sloj, kernel_initializer='he_normal')(m)
    m = layers.BatchNormalization()(m)
    m = layers.Activation('relu')(m)
m = layers.Dense(embedding, activation='relu', kernel_initializer='he_normal')(m)
movie_embedding = L2NormalizeLayer()(m)

# dot produkt i izlaz
similarity = layers.Dot(axes=1, name='cosine_similarity')([user_embedding, movie_embedding])
output = SqueezeLayer()(similarity)

model = tf.keras.Model(inputs=[user_input, movie_input], outputs=output)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae', 'mse'])

model.summary()

data_dev = data.skip(train_size).take(dev_size)

data_dev = data_dev.map(lambda x, y: ((norm_user(x[0]),tf.concat([tf.cast(norm_movies(x[1][:, :3]), tf.float32),tf.cast(x[1][:, 3:], tf.float32)], axis=1)),scale_y(y))).prefetch(tf.data.AUTOTUNE)


history = model.fit(data_train, validation_data=data_dev, epochs=5, steps_per_epoch=train_size, validation_steps=dev_size)


model.save(f'model_128_128_64_32.keras')
joblib.dump(history, 'histori_128_128_64_32.pkl')

test_score = model.evaluate(data_test)

model.summary()

movies_net = tf.keras.Model(inputs=movie_input, outputs=movie_embedding)
movies_net = movies_net.predict(X_movie)

data_test = data.skip(train_size).take(test_size)


data_test = tf.data.Dataset.from_generator(lambda: batch_generator(ratings_path= csv_ratings, movies_path= csv_movies,batch_size= 100, train = False), output_signature= ((tf.TensorSpec(shape=(None, 22), dtype=tf.float64, name = 'user'), tf.TensorSpec(shape=(None, 25), dtype=tf.float64, name = 'movie')), tf.TensorSpec(shape=(None,3), dtype=tf.float32)))

norm_user.mean
norm_user.variance
norm_movies.mean
norm_movies.variance

t = StandardizationLayer()
t.set_weights(norm_user.mean)

m[:,:5]
(u,m), y = next(iter(data_test))
data_test = data_test.map(lambda x, y: ((norm_user(x[0]),tf.concat([tf.cast(norm_movies(x[1][:, 2:5]), tf.float32),tf.cast(x[1][:, 3:], tf.float32)], axis=1)),scale_y(y))).repeat().prefetch(tf.data.AUTOTUNE)


m_net = model.movie_net


X_user_id, maska ,y_id = scale(df, user, movies_feat, user_id = 28)
u_id = X_user_id[0,0]
X_user_id = X_user_id[0:1,1:]  ##SVAKI RED JE ISTI, A PRVA KOL JE USER_ID, MORA 0:1 DA BI SHAPE BIO (1,-1)

u_net = model.user_net
u_embed = u_net.predict(X_user_id)
print(u_embed.shape)

pred = tf.linalg.matmul(u_embed, m_embed, transpose_b= True)
pred_negledani = tf.boolean_mask(pred, ~maska, axis = 1)
val, idx = tf.math.top_k(pred_negledani, k = 10)

