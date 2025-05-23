from prep import *

import tensorflow as tf
from model import ColaborativeFiltering
from tensorflow import keras
import joblib
import matplotlib.pyplot as plt
ratings, movies = read_data_lake()
user, movies, df = prep_pipeline(ratings, movies)

# pl.read_database(query='SELECT * FROM data_lake.ratings', connection=conn)

# engine = create_engine(f"postgresql+psycopg2://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/movie_recommendation")
# conn = engine.connect()


X_user_full, _, _, _ = NN_prep(df, user, movies)
X_user, X_movie, y, scalers = NN_prep(df, user, movies, user_id = 103)

model = keras.models.load_model('model_proba.keras')
history = joblib.load('history_proba.pkl')

mse = pl.DataFrame({'mse': history.history['mse'], 'val_mse': history.history['val_mse']})
plt.figure(figsize = (6,4))
plt.plot(mse['mse'], label='loss')
plt.plot(mse['val_mse'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()
X_user = X_user[0]

user_vecs = tf.repeat(tf.reshape(X_user, (1, -1)), tf.shape(X_movie)[0], axis=0)
preds = model.predict([user_vecs, X_movie])
preds[np.argmax(preds)]
preds.shape
X_user

model.recommend(X_user, X_movie)
print(X_movie.shape, X_user.shape)




proba = user[100]
df.select(user.select(pl.exclude('userid')).columns)
df.select(movies.select(pl.exclude(['movieid','title'])).columns + ['#ratings_film'])

user_proba, movies_proba, y_proba, scalers_proba = NN_prep(df, proba, movies, user_id = proba[0,0])
user_proba = user_proba[0]
user_vec = user_proba

movies

X_user
X_user
user_vecs = tf.repeat(tf.reshape(X_user, (1, -1)), tf.shape(X_movie)[0], axis=0)
preds = model.predict([user_vecs, X_movie])
preds.shape
np.argmax(preds)
film = df[798612]['movieid']
movies.filter(pl.col('movieid') == film)

df.filter(pl.col('userid') == 103).sort('rating')


# user_vec = X_user[dev_idx[10]]
# k = 10
# # movie_titles = None
# preporuke = model.recommend(user_vec, X_movie, user_seen_movie_indices, k=k)#, movie_titles=movie_titles)
# print('Preporuke za usera', user, ':', preporuke)

# Pretpostavljamo da je df DataFrame iz prep_pipeline
# lista_idx = [i[0] for i in preporuke]
# lista_id = df['movieid'].to_list()  # ili movie_features['movieid'].to_list() ako koristiš movie_features
# preporuceni_movieid = [lista_id[idx] for idx in lista_idx]

# lista_id = [i[0] for i in preporuke]


# movies.filter(pl.col('movieid') == 54900)
# df.filter(pl.col('movieid') == 54900)
# df
# X_movie
# ratings.filter(pl.col('movieid') == 57480)

# X_movie
# movies
# model.summary()
# type(model)
# proba = ratings.pivot(values= 'rating', index= 'userid', on = 'movieid')
# proba = df[:,:3]
user[0]


user_id = 100
df.filter((pl.col('userid') == user_id) & (pl.col('movieid') == 1090))
ratings.filter((pl.col('userid') == user_id) & (pl.col('movieid') == 1090))
gledani_movieid = set(df.filter(pl.col('userid') == user_id)['movieid'].to_list())
movieid_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies['movieid'].to_list())}

###PROVERI DA LI RADI PRE NEGO STO SE IZBACE SEEN MOVIES A ONDA IZBACI SEEN MOVIES!!!

user_vec = X_user[dev_idx[10]]
movie_matrix = X_movie
tf.concat

user_vecs = tf.repeat(tf.reshape(user_vec, (1, -1)), tf.shape(movie_matrix)[0], axis=0)
preds = model.predict([user_vecs, movie_matrix])
preds.shape
# mask_indices = tf.constant(list(user_seen_movie_indices), dtype=tf.int32)
# preds = tf.tensor_scatter_nd_update(
#     tf.squeeze(preds),
#     tf.expand_dims(mask_indices, 1),
#     tf.fill([tf.size(mask_indices)], tf.constant(-float('inf'), dtype=preds.dtype))
# )
top_k_idx = tf.argsort(preds, direction='DESCENDING')[:k]
if movie_titles is not None:
    return [(movie_titles[int(i)], float(preds[i])) for i in top_k_idx]
else:
    return [(int(i), float(preds[i])) for i in top_k_idx]





















