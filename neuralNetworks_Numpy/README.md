## Neural Network from Scratch using NumPy with Custom Logit Class

This project explores the fundamentals of building neural networks manually using NumPy, emphasizing a clear understanding of forward propagation, softmax-based classification, and gradient calculations without relying on high-level libraries such as TensorFlow or PyTorch.

At the core of the project is a custom `Logit` class, initially designed for logistic regression, now enhanced to support multilayer neural networks with various activation functions (sigmoid, softmax) and L1/L2 regularization methods. The class also includes early stopping functionality, learning rate scheduling, and weight tracking to monitor training progress.

### Dataset
The model is trained on the MNIST dataset (70,000 handwritten digits), prepared as follows:
- Loaded via `fetch_openml("mnist_784")`
- Normalized manually (pixel values scaled by 255)
- Labels converted to one-hot encoding
- Dataset split into: 50,000 training / 10,000 validation (dev) / 10,000 testing

### Architecture
- Customizable multilayer neural network (e.g., input → hidden layers → output)
- Weight matrices initialized manually using `np.random.rand(...) - 0.5`
- Forward propagation using custom `sigmoid` and `softmax` methods from `Logit`
- Supports numerical gradient calculation for correctness verification
- Loss function: cross-entropy with optional L1/L2 regularization
- Performance evaluation: accuracy on training, validation, and test datasets

### Highlights
- Entire neural network logic implemented purely with NumPy
- Flexible architecture allowing multiple hidden layers
- Reused and extended a custom logistic regression (`Logit`) class
- Integrated L1/L2 regularization
- Structured pipeline: preprocessing, encoding targets, training, evaluation
