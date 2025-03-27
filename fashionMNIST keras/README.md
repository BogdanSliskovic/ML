## Fashion MNIST – Image Classification with Neural Networks

This project explores the Fashion MNIST dataset, a well-known benchmark in image classification containing grayscale images of 10 different clothing categories. Each sample is a 28x28 pixel image, and the goal is to correctly classify it into its corresponding class.

The dataset was preprocessed by normalizing pixel values and reshaping input arrays. I experimented with different neural network architectures using TensorFlow and Keras, focusing on the effects of:
- Number of hidden layers and neurons
- Activation functions (ReLU, sigmoid)
- Dropout regularization
- Batch size and number of epochs

I visualized class samples to better understand the input data and used metrics like accuracy, precision, and recall to evaluate model performance. The final model achieves solid classification results and serves as a foundation for future improvements such as convolutional layers or data augmentation.

### Key Concepts:
- Image preprocessing and reshaping
- Building fully connected neural networks with Keras
- Regularization via Dropout
- Hyperparameter tuning (epochs, batch size, optimizers)
- Performance evaluation on training and validation sets
