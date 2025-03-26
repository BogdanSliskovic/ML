## Logistic Regression from Scratch using NumPy

In this project, I built a custom `Logit` class using pure NumPy, without relying on Scikit-learn for modeling. The class supports both **binary** and **multiclass** classification (using sigmoid and softmax activations), as well as L1/L2 regularization, batch gradient descent, learning rate decay, early stopping, and automatic weight rollback.

The model was applied to the MNIST dataset. In the binary setting, the goal was to distinguish one digit from the others. In the multiclass setting, all 10 digits were included using softmax activation and one-hot encoded labels.

All features were normalized, and a bias term was manually added. Early stopping was implemented to monitor development set performance and revert to the best weights if overfitting occurred. I also implemented a custom `fitReg()` method for lambda grid search (regularization strength), and used `joblib` to save the best models for future use.

---

### Key Features
- Logistic regression built from scratch using NumPy
- Support for both binary and multiclass classification (sigmoid / softmax)
- L1 and L2 regularization with custom grid search (`fitReg`)
- Batch gradient descent with learning rate decay
- Early stopping with rollback to best weights
- Performance tracking on training and dev sets
- Model persistence using `joblib`

---

### Binary Classification – MNIST

| Model                | Regularization | Train Accuracy | Dev Accuracy | Test Accuracy |
|---------------------|----------------|----------------|--------------|----------------|
| Logit               | None           | 0.9642         | 0.9628       | 0.9626         |
| Logit               | L1             | 0.9651         | 0.9634       | 0.9635         |
| Logit               | L2             | 0.9641         | 0.9629       | 0.9625         |
| Sklearn LogisticReg | None           | 0.9782         | 0.9732       | 0.9745         |
| Sklearn LogisticReg | L1             | 0.9782         | 0.9732       | 0.9752         |
| Sklearn LogisticReg | L2             | 0.9775         | 0.9732       | 0.9735         |

---

### Multiclass Classification – All MNIST Digits

Each softmax model below uses a different initial learning rate. All models were trained using the same `Logit` class, extended to support softmax activation for multiple classes.

| Model      | Initial LR | Train Accuracy | Dev Accuracy | Test Accuracy |
|------------|------------|----------------|--------------|----------------|
| softmax    | 0.1        | 0.9005         | 0.8934       | 0.8977         |
| softmax2   | 1.0        | 0.9251         | 0.9151       | 0.9190         |
| softmax3   | 0.5        | 0.9210         | 0.9125       | 0.9161         |
| sklearn    | —          | 0.9448         | 0.9148       | 0.9139         |

The custom softmax models show solid performance, with noticeable improvements as the learning rate is tuned. The best result was achieved by `softmax2` with an initial learning rate of 1.0, outperforming the other custom variants **as well as the Scikit-learn implementation** on the test set.
