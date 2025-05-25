import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

class CollaborativeFilteringJAX(nn.Module):
    user_layers: Sequence[int]
    movie_layers: Sequence[int]
    embedding_dim: int = 32

    @nn.compact
    def __call__(self, user_x, movie_x):
        # User tower
        for units in self.user_layers:
            user_x = nn.Dense(units)(user_x)
            user_x = nn.tanh(user_x)
        user_emb = nn.Dense(self.embedding_dim)(user_x)
        user_emb = user_emb / (jnp.linalg.norm(user_emb, axis=-1, keepdims=True) + 1e-8)

        # Movie tower
        for units in self.movie_layers:
            movie_x = nn.Dense(units)(movie_x)
            movie_x = nn.tanh(movie_x)
        movie_emb = nn.Dense(self.embedding_dim)(movie_x)
        movie_emb = movie_emb / (jnp.linalg.norm(movie_emb, axis=-1, keepdims=True) + 1e-8)

        # Cosine similarity
        sim = jnp.sum(user_emb * movie_emb, axis=-1, keepdims=True)
        return sim