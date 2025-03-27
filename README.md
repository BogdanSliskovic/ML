# Machine Learning Projects – Bogdan Slišković

Welcome to my machine learning project collection!  
This repository is organized into several subfolders, each containing a self-contained project with its own Jupyter notebooks and local README files.

---

## Contents

### LogisticRegression_Numpy(./LogisticRegression_Numpy)
Custom `Logit` class implemented in pure NumPy, supporting both binary and multiclass classification. Features include L1/L2 regularization, batch gradient descent with learning rate decay, early stopping, and model rollback.  
Applied to the MNIST dataset, achieving competitive results versus Scikit-learn baselines. Also includes custom grid search and model saving with `joblib`.

### neuralNetworks_Numpy(./neuralNetworks_Numpy) `in progress`
Manual implementation of a neural network using only NumPy. Built on top of the `Logit` class to support forward propagation and softmax-based classification. Uses MNIST, with manual data preprocessing, one-hot encoding, and full forward pass logic.  
Lays the foundation for backpropagation and hidden layer support in future iterations.

### [fashionMNIST_keras](./fashionMNIST_keras) `in progress`
Image classification using fully connected neural networks built with TensorFlow and Keras. Applied on the Fashion MNIST dataset with exploration of activation functions, dropout regularization, and tuning hyperparameters.  
Good foundation for future CNN implementation.

### [Titanik](./Titanik)
Comprehensive end-to-end modeling on the Titanic dataset with scikit-learn. Includes feature engineering (`deck-class`, family size), missing value imputation via KNN, model tuning with `RandomizedSearchCV`, and dimensionality reduction via PCA.  
Also explores unsupervised learning by adding `KMeans` and `DBSCAN` cluster features into the modeling pipeline.

### [creditRiskModeling](./Kredit)
Binary classification project using a credit dataset from OpenML. Models include Logistic Regression, SVM, and Random Forest, each with a tailored preprocessing pipeline.  
Includes analysis of overfitting, model evaluation using F1-score, and advanced interpretation tools like SHAP values and feature importances.

### [irisClassifiers](./IRIS)
Clean, minimal exploration of the Iris dataset focusing on decision boundaries and model behavior.  
Compares Logistic Regression, SVM, and Random Forest classifiers while highlighting signs of overfitting and underfitting.

### [git automation scripts](./gitExe)
Two Python scripts that automate Git tasks: one for checking status and pulling updates, the other for add/commit/push with a timestamp.  
Designed for convenience and speed in daily Git workflows.

---

## About

Each project folder contains:
- Jupyter notebooks
- A local `README.md` with full explanation
- Clean structure for reproducibility and experimentation

---

## Contact

**Bogdan Slišković**  
bogdansliskovic@gmail.com  
https://github.com/BogdanSliskovic  
https://www.linkedin.com/in/bogdan-sliskovic/
