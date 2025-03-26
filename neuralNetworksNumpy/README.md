## Neural Network Foundations using NumPy and Custom Logistic Regression

In this project, I use a custom `Logit` class (previously implemented from scratch using NumPy) as the foundation for building a simple neural network architecture.

The goal is to explore how data flows through a neural network at a low level, without relying on deep learning libraries like TensorFlow or PyTorch. This includes manual implementation of forward propagation, activation functions, backpropagation and training.

The model is trained on the MNIST dataset (handwritten digit classification). Features are normalized and one-hot encoded targets are used for multiclass prediction.

### Structure
- The first hidden layer is created manually using NumPy and applies a sigmoid activation.
- The `Logit` class is then used as the output layer with softmax activation.
- Different weight initializations and learning rates are tested.
- Forward pass is implemented manually.
- Model evaluation is based on accuracy on train/dev/test sets.

### Key Concepts:
- Forward propagation using matrix operations
- Custom activation flow through multiple layers
- Using a hand-built logistic regression class as the final layer
- Preparing structure for gradient-based training
- Applied on MNIST dataset with multiclass targets

### Next Steps
The current version performs forward propagation only. Backpropagation and full training through multiple layers will be implemented next.


