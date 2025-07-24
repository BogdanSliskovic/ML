from prep import *
from model import *
import tensorflow as tf
from tensorflow import keras
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
model = keras.models.load_model("model_128_128_64_32.keras", custom_objects={
    "L2NormalizeLayer": L2NormalizeLayer,
    "SqueezeLayer": SqueezeLayer,
    "MovieRecommender": MovieRecommender
})
norm_user = keras.models.load_model('user_scaler.keras')
norm_movies = keras.models.load_model('movies_scaler.keras')
model.summary()
model.user_net.summary()
model.movie_net.summary()

csv_movies = '../ml-32m/movies.csv'
csv_ratings = '../ml-32m/ratings.csv'
user_kolone, movies_kolone = imena_kolona(csv_ratings, csv_movies)
history = joblib.load('histori_128_128_64_32.pkl')
norm_user = keras.models.load_model('user_scaler.keras')
norm_movies = keras.models.load_model('movies_scaler.keras')
hist = pl.DataFrame(history.history)

ratings_test = pl.read_csv('../ml-32m/ratings_test.csv')
movies = pl.read_csv('../ml-32m/movies.csv')
user, movies, y  = prep_pipeline(ratings_test, movies)

(user[:,2:],movies[:,2:]), y
model.evaluate((user[:,2:],movies[:,2:]), y[:,2:3])

movies = pl.read_csv('../ml-32m/movies.csv')

data_test = tf.data.Dataset.from_generator(lambda: batch_generator(ratings_path= '../ml-32m/ratings_test.csv', movies_path= csv_movies,batch_size= 2**16, train = True),
    output_signature= ((tf.TensorSpec(shape=(None, 20), dtype=tf.float64, name = 'user'), tf.TensorSpec(shape=(None, 23),
    dtype=tf.float64, name = 'movie')), tf.TensorSpec(shape=(None,1), dtype=tf.float32))).prefetch(tf.data.AUTOTUNE)
next(iter(data_test))

data_test = data_test.map(lambda x, y: ((norm_user(x[0]),tf.concat([tf.cast(norm_movies(x[1][:, :3]), tf.float32),tf.cast(x[1][:, 3:], tf.float32)], axis=1)),scale_y(y))).prefetch(tf.data.AUTOTUNE)

y_test = scale_y(y[:,2].to_numpy())

y_test_pred = model.predict(data_test)

mean_absolute_error(y_test, y_test_pred)
mean_squared_error(y_test, y_test_pred)

data_test.map(lambda x,y: x[0])
data_train = tf.data.Dataset.from_generator(lambda: batch_generator(ratings_path= '../ml-32m/ratings_train.csv', movies_path= csv_movies,batch_size= 2**16, train = False),
    output_signature= ((tf.TensorSpec(shape=(None, 22), dtype=tf.float64, name = 'user'), tf.TensorSpec(shape=(None, 25),
    dtype=tf.float64, name = 'movie')), tf.TensorSpec(shape=(None,3), dtype=tf.float32))).prefetch(tf.data.AUTOTUNE)
next(iter(data_train))
i = 0


#PROLAZI KROZ CEO TRAIN DATASET I PRAVI MOVIE MATRICU SA PRIMARY KLJUCEM MOVIEID (INACE JE (USERID,MOVIEID) PA IMA 31M PODATAKA, OVAKO IMA 85 HILJADA RAZLICITIH FILMOVA PA JE 85000X25 UMESTO 31MX25
# for (u,m),y in data_train:
#     if i == 0:
#         X_movies = pl.DataFrame(m.numpy(),schema= movies_kolone).with_columns(
#             pl.col("movieId").cast(pl.Int64))
#     else:
#         df = pl.DataFrame(m.numpy(), schema=movies_kolone).with_columns(pl.col("movieId").cast(pl.Int64))
#         X_movies = pl.concat([X_movies, df])
#         X_movies = X_movies.unique(subset="movieId", keep="first")
#     print(i)
#     i +=1
#     print(X_movies.shape)
#     if X_movies.height == movies.height:
#         break
# X_movies.write_csv('../ml-32m/X_movies.csv')
X_movies_id = pl.read_csv('../ml-32m/X_movies.csv')
def spoji_users(df):
    # df shape: (batch, 25)
    df = tf.cast(df, tf.float32)
    return tf.concat(
        [df[:, :2], norm_user(df[:, 2:])],
        axis=1
    )

def spoji_movies(df):
    # df shape: (batch, 22)
    df = tf.cast(df, tf.float32)
    return tf.concat(
        [df[:,:2], norm_movies(df[:, 2:5]), df[:, 5:]],
        axis=1
    )

def spoji_labels(df):
    # df shape: (batch, 3)
    df = tf.cast(df, tf.float32)
    return tf.concat(
        [df[:, :2], scale_y(df[:, 2:])],
        axis=1
    )

data_train = data_train.map(
    lambda x, y: (
        (spoji_users(x[0]), spoji_movies(x[1])),
        spoji_labels(y)
    )
).prefetch(tf.data.AUTOTUNE)

data_train = data_train.map(lambda x, y: ((norm_user(x[0]),tf.concat([tf.cast(norm_movies(x[1][:, :3]), tf.float32),tf.cast(x[1][:, 3:], tf.float32)], axis=1)),scale_y(y))).repeat().prefetch(tf.data.AUTOTUNE)

data_test = tf.data.Dataset.from_generator(lambda: batch_generator(ratings_path= '../ml-32m/ratings_test.csv', movies_path= '../ml-32m/movies.csv',batch_size= 10000, train = False), output_signature= ((tf.TensorSpec(shape=(None, 22), dtype=tf.float64, name = 'user'), tf.TensorSpec(shape=(None, 25), dtype=tf.float64, name = 'movie')), tf.TensorSpec(shape=(None,3), dtype=tf.float32)))
next(iter(data_test))
model.evaluate(data_test)


userId = 29499
temp = test.filter(pl.col('userId') == userId)
gledani = temp['movieId'].to_list()
u,m, y = prep_pipeline(temp, movies)

st = (u[:, 2:].to_numpy() - norm_user.mean.numpy()) / np.sqrt(norm_user.variance)
st[:,0] = 0
norm_movies.mean.numpy()

#IZBACUJEM ID KOLONE
X_movies = X_movies_id[:,2:]
norm_movies.mean.numpy()
norm_movies.variance.numpy()

num_movies = X_movies[:,:3]
st_movies = (num_movies.to_numpy() - norm_movies.mean.numpy()) / np.sqrt(norm_movies.variance)
st_movies = pl.DataFrame(st_movies, schema=X_movies.columns[:3])
X_movies = pl.concat([st_movies, X_movies[:,3:]], how="horizontal")
embed_u = model.user_net.predict(st)
embed_m = model.movie_net.predict(X_movies.to_numpy())
embed_m[0]
embed_u = tf.repeat(embed_u[0:1], embed_m.shape[0], axis = 0)
embed_u.shape, embed_m.shape

dot = tf.reduce_sum(embed_u * embed_m, axis=1)
temp['rating']
tf.math.top_k(dot, k=10)[1]
X_movies_id[10158]
tf.reduce_max(dot)



m.filter(pl.col('moviId') == m.select(pl.col('movieId')).unique())
m.is_duplicated()


model.user_net.summary()
(u,m), y = next(iter(data_test.map(uzmi_usera(54488))))

tf.boolean_mask(u[:,2:], tf.equal(u[:,0], 54488))[0]
tf.where(u[:,0] == 54488)
t = u[u[:,0] == 54488]
odgledani_filmovi = t[:,1]
odgledani_filmovi
model.user_net.predict(t[0,2:])
model.user_net.predict(t[0:1,2:])
t[0,2:]
t[0:1,2:]
tf.boolean_mask(u[:,0] == 54488)

embed_m = model.movie_net.predict(data_test.map(lambda x,y: x[1][:,2:]))
embed_m.shape
suma = 0
sum([i**2 for i in embed_m[100]])
embed_m.shape

userId = 54488
def uzmi_usera(userId):
    def fn(x,y):
        user = x[0]
        return tf.boolean_mask(user[:,2:], tf.equal(user[:,0], userId))[0:1]
    return fn
embed_u = model.user_net.predict(data_test.map(uzmi_usera(userId)))
embed_u[5,:5]
tf.repeat(embed_u[0:1], embed_m.shape[0], axis = 0)


u
t = data_test.map(lambda x,y: x[1][:,2:])
hist = pl.DataFrame(history.history)
plt.style.use('ggplot')
plt.figure(figsize=(8,5))
plt.plot(hist["loss"], label="Train Loss")
plt.plot(hist["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

m_net = model.movie_net

m_embed = m_net.predict(X_movie)

X_user_id, maska ,y_id = scale(df, user, movies_feat, user_id = 28)
u_id = X_user_id[0,0]
X_user_id = X_user_id[0:1,1:]  ##SVAKI RED JE ISTI, A PRVA KOL JE USER_ID, MORA 0:1 DA BI SHAPE BIO (1,-1)

u_net = model.user_net
u_embed = u_net.predict(X_user_id)
print(u_embed.shape)

pred = tf.linalg.matmul(u_embed, m_embed, transpose_b= True)
pred_negledani = tf.boolean_mask(pred, ~maska, axis = 1)
val, idx = tf.math.top_k(pred_negledani, k = 10)












### inference.py
m_net = model.movie_net
m_embed = m_net.predict(X_movie)
print(m_embed.shape)

X_user_id, _, y_id, _ = scale(df, user, movies_feat, user_id = 28)
u_id = X_user_id[0,0]
m_id = 
X_user_id = X_user_id[0,1:]  ##SVAKI RED JE ISTI, A PRVA KOL JE USER_ID

u_net = model.user_net
u_embed = u_net.predict(tf.expand_dims(X_user_id,0))
print(u_embed.shape)

pred = tf.linalg.matmul(u_embed, m_embed, transpose_b= True)
val, idx = tf.math.top_k(pred, k = 10)
tf.gather(pred, idx, axis = 1) == val













