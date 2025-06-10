# Movie Recommendation System  
![Architecture](Train.drawio.png)

This project implements a scalable movie recommendation system using PostgreSQL for data storage, Polars for efficient data processing and TensorFlow for batch processing and model training. The dataset used is the MovieLens dataset, containing 32 million movie ratings.  
The model is trained on 20 million ratings, while the remaining data is continuously injected into production on a daily basis.

The goal of this project is to simulate a production-ready architecture featuring an end-to-end ETL pipeline and automation with Apache Airflow. The system is designed to be modular, scalable, and maintainable, enabling continuous learning and adaptation of the model to new data and evolving user behavior.

---

## Preparing data for training

Functions for batch data processing, reading from and writing to the PostgreSQL database, as well as data scaling and feature engineering, are implemented in [prep.py](./prep.py).  
This module handles extraction of from raw data, transformation into feature-rich user and movie tables, and normalization of numeric features to prepare the dataset for model training.

---

## Model Architecture

The model is implemented in [model.py](./model.py) and is structured as a custom Keras class.

It uses a two-tower architecture:

- **User Net**: A sequential neural network that transforms user features into a dense embedding space.  
- **Movie Net**: A parallel network that does the same for movie features.

Both towers output 64-dimensional embeddings (or another size depending on the embedding parameter), which are L2-normalized and combined using a dot product layer to approximate cosine similarity.  
This value serves as the predicted user–movie interaction (e.g., rating score).

The model is compiled using the Nadam optimizer with mean squared error (MSE) as the loss function and includes support for:

- Customizable layer sizes via `user_layers` and `movie_layers`  
- Optional regularization via `user_reg` and `movie_reg`  
- Full serialization support through `get_config()` and `from_config()` methods

This setup allows flexible experimentation with embedding dimensions, learning rates, and regularization, making it suitable for iterative training in a production pipeline.

---

## Model training

Model training is handled in [train.py](./train.py).  
It brings together all components of the pipeline—data loading, preprocessing, and model optimization—into a single training routine.

### Key steps include:

- **Data pipeline**:  
  A custom `batch_generator` streams training data into TensorFlow using `tf.data.Dataset.from_generator`.  
  Since the full 20 million rating records cannot fit into memory, this streaming approach allows efficient batch-wise training by loading only a portion of the data at a time.  
  This design enables scalable model training without requiring large-memory infrastructure.

- **Feature scaling**:  
  Once the data is transformed, it is loaded into the `data_storage` schema.  
  From there, column-wise averages and standard deviations are computed using SQL queries and Polars.  
  These statistics are used to normalize the numeric features before feeding them into the model.

- **Model setup**:  
  The `ColaborativeFiltering` model is initialized with configurable architecture and trained using a validation set and callbacks such as `EarlyStopping` and `ReduceLROnPlateau` to ensure optimal convergence.

The training script reflects production-minded practices, including dynamic scaling, robust checkpointing, and modular component design.  
Model is saved and passed into [inference.py](./inference.py).

---

## Inference

Model inference is implemented in [inference.py](./inference.py).  
It uses the trained `ColaborativeFiltering` model to generate top-N movie recommendations for a given user.

### Key steps:

- **Precomputed movie embeddings**:  
  The movie tower (`model.movie_net`) is used once to generate and store embeddings for all movies.  
  These embeddings are reused across all users, which significantly improves inference speed and reduces redundant computation.

- **On-demand user embedding**:  
  For any given user, the user tower (`model.user_net`) generates a single user embedding vector on the fly.

- **Dot product scoring**:  
  The user's embedding is compared against all precomputed movie embeddings using a dot product, representing predicted affinity.  
  Movies the user has already seen are masked out.

- **Top-N selection**:  
  The system returns the top 10 highest-scoring unseen movies using `tf.math.top_k()`.

This approach enables fast, real-time inference, as only the user embedding needs to be computed per request, while the movie embeddings can be cached and reused indefinitely (or until retraining).
