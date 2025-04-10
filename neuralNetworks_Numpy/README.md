## Neural Network from Scratch using NumPy with Custom Logit Class

This project explores the basics of building a neural network manually using NumPy, with a focus on understanding forward propagation and softmax-based classification — without relying on high-level libraries like TensorFlow or PyTorch.

At the core of the project is a custom `Logit` class, originally designed for logistic regression, which has been extended to support softmax activation for multiclass problems. The class also includes features like L1/L2 regularization, early stopping, and weight tracking.

### Dataset
The model is trained on the full MNIST dataset (70,000 handwritten digits), using the following setup:
- `fetch_openml("mnist_784")` for data loading
- Manual normalization (pixel values / 255)
- One-hot encoding of labels
- Data split: 50,000 train / 10,000 dev / 10,000 test

### Architecture
- The neural net currently has a single layer (input → output)
- Weight matrix is initialized manually: `w = np.random.rand(...) - 0.5`
- Forward pass is implemented using softmax: `Logit.softmax(x @ w)`
- Loss: cross-entropy
- Evaluation: classification accuracy on train/dev/test sets

### Highlights
- All logic implemented in NumPy
- Full forward pass with softmax for multiclass classification
- Custom logistic regression class reused and extended
- Foundation laid for implementing backpropagation later
- Clean manual pipeline: feature prep, target encoding, training, evaluation

### Next Steps
- Implement backpropagation manually
- Add support for hidden layers and different activations
- Track loss and accuracy over iterations
