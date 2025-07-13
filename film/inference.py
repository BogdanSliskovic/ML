from prep import *
from model import *
import tensorflow as tf
from tensorflow import keras
import joblib
import matplotlib.pyplot as plt
import numpy as np
model = keras.models.load_model("proba.keras", custom_objects={
    "L2NormalizeLayer": L2NormalizeLayer,
    "SqueezeLayer": SqueezeLayer,
    "MovieRecommender": MovieRecommender
})

raw_dataset = tf.data.TFRecordDataset('dataset.tfrecord')


history = joblib.load('histori_128_128_64_32.pkl')
norm_user = keras.models.load_model('norm_user.keras')
norm_movies = keras.models.load_model('norm_movies.keras')
hist = pl.DataFrame(history.history)

test = pl.read_csv('../ml-32m/ratings_test.csv')
movies = pl.read_csv('../ml-32m/movies.csv')
data_test = tf.data.Dataset.from_generator(lambda: batch_generator(ratings_path= "../ml-32m/ratings_test.csv", movies_path= '../ml-32m/movies.csv',batch_size= 10000, train = False), output_signature= ((tf.TensorSpec(shape=(None, 22), dtype=tf.float64, name = 'user'), tf.TensorSpec(shape=(None, 25), dtype=tf.float64, name = 'movie')), tf.TensorSpec(shape=(None,3), dtype=tf.float32)))
userId = 54488
temp = test.filter(pl.col('userId') == userId)
u,m, y = prep_pipeline(temp, movies)
norm_user.variance.numpy()
u[0, 2:] - norm_user.mean.numpy()
temp.mean()['rating']
st = (u[0, 2:].to_numpy() - norm_user.mean.numpy()) / np.sqrt(norm_user.variance)
st[0,0] = -100

model.user_net(st)
model.movie_net

m.filter(pl.col('moviId') == m.select(pl.col('movieId')).unique())
m.is_duplicated()

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

data_test = data_test.map(
    lambda x, y: (
        (spoji_users(x[0]), spoji_movies(x[1])),
        spoji_labels(y)
    )
).prefetch(tf.data.AUTOTUNE)
(u,m), y = next(iter(data_test))
user = u


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



















tf.random.set_seed(42)
lista_id = tf.convert_to_tensor(df.select(pl.col('userid').unique()).to_numpy().flatten(), dtype=tf.float32)
user_ids = tf.random.shuffle(lista_id)[:5]

X_user_full, X_movie_full ,y_full, scalers_full,  = scale(df, user, movies)
X_user_id, X_movie, y, scalers = scale(df, user, movies, user_id = user_ids)
X_user = X_user_id[:,1:]
assert [i for i in scalers if not np.allclose(scalers[i], scalers_full[i])] == []
assert X_user.shape[0] == X_movie.shape[0] == y.shape[0]


user_vec = X_user_id[0, 1:]  # ili kako već uzimaš vektor korisnika
user_vecs = tf.repeat(tf.reshape(user_vec, (1, -1)), X_movie_full.shape[0], axis=0)
preds = model.predict([user_vecs, X_movie_full])

# Izbaci filmove koje je korisnik već gledao:
gledani_movieid = set(df.filter(pl.col('userid') == user_id)['movieid'].to_list())
svi_movieid = X_movie_full[:, 0].numpy()  # pretpostavka: prva kolona je movieid
mask = np.array([mid not in gledani_movieid for mid in svi_movieid])
preds_novi = preds[mask]
movieid_novi = svi_movieid[mask]


model.movie_net



mse = pl.DataFrame({'mse': history.history['mse'], 'val_mse': history.history['val_mse']})
plt.figure(figsize = (6,4))
plt.plot(mse['mse'], label='loss')
plt.plot(mse['val_mse'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()


model.summary()
preds = model.predict([X_user, X_movie])
preds.shape
prvi = int(tf.argmax(preds, axis=0))
rez = preds[prvi]
film_id = int(X_user_id[prvi][0])
df.filter(pl.col('movieid') == film_id).select(['userid', 'movieid', 'rating'])

movies.select(['movieid', 'title']).join(
    df.filter(pl.col('userid') == id).select(['userid', 'movieid', 'rating']),
    on='movieid',
    how='inner'
)

movies.filter(pl.col('movieid') == film_id).select(['movieid', 'title'])

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













