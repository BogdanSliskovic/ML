from prep import *

import tensorflow as tf
from model import ColaborativeFiltering
from tensorflow import keras
import joblib
import matplotlib.pyplot as plt
ratings, movies = read_data_lake()
user, movies, df = prep_pipeline(ratings, movies)
model = keras.models.load_model('model_proba.keras')
history = joblib.load('history_proba.pkl')

# pl.read_database(query='SELECT * FROM data_lake.ratings', connection=conn)

# engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
# conn = engine.connect()





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













