# Movie Recommendation System

![Architecture](Train.drawio.png)

This project implements a scalable and efficient movie recommendation system using the MovieLens dataset containing 32 million movie ratings. It leverages **Polars** for fast data processing and **TensorFlow** with Keras for model training and inference.

---

## Overview

The system uses a **two-tower neural network architecture** to model user preferences and movie features separately, generating embeddings for each and predicting their affinity via cosine similarity. The pipeline is designed for large-scale datasets with batch-wise streaming and modular components for training and inference.

---

## Data Preparation

Data processing and feature engineering are implemented in [`prep.py`](./prep.py). Key steps include:

- Loading raw CSV data of user ratings and movie metadata.
- Extracting and encoding movie genres as one-hot dummy variables.
- Computing per-user average ratings per genre as features.
- Creating two main dataframes:
  - **User DataFrame**: 22 columns including user and movie IDs plus average genre ratings.
  - **Movie DataFrame**: 25 columns including IDs, number of ratings, average rating, release year, and genre dummy variables.

These feature matrices serve as input to the model.

---

## Model Architecture

Defined in [`model.py`](./model.py), the model consists of:

- **User Network**: A sequential feed-forward neural network that transforms user features into a 32-dimensional embedding vector.
- **Movie Network**: A parallel network that transforms movie features into an embedding of the same dimension.

Both embeddings are L2-normalized to ensure unit length, enabling the dot product layer to compute cosine similarity as the interaction score between user and movie embeddings.

The output represents the predicted user rating affinity on a scale from -1 to 1.

The model supports:

- Customizable hidden layer sizes.
- Full serialization and deserialization via `get_config()` and `from_config()` methods.

---

## Training

Training is orchestrated in [`train.py`](./train.py), featuring:

- **Batch Streaming**: Using a custom `batch_generator` and TensorFlow's `tf.data.Dataset` to load and process data in batches, enabling training on datasets too large to fit into memory.
- **Feature Normalization**: Numerical features are standardized via Keras normalization layers integrated directly in the training pipeline.
- **Rating Scaling**: Target ratings are scaled to [-1, 1] range to align with the cosine similarity output.
- **Model Optimization**: Using Adam optimizer and Mean Squared Error loss.

The trained model is saved after training for later use in inference.

---

## Inference

Implemented in [`inference.py`](./inference.py), the inference pipeline optimizes prediction speed by:

- **Precomputing movie embeddings** once and caching them for reuse.
- **Generating user embeddings on demand** for each input user.
- **Calculating dot product scores** between user embedding and all movie embeddings to estimate preferences.
- **Masking out already rated movies** to avoid recommending seen content.
- **Returning top-N recommendations**, typically the top 10 unseen movies.

This design supports efficient real-time recommendations.
