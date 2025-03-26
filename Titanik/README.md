## 🛳️ Titanic Survival Prediction with Advanced Modeling & Clustering

This project explores the Titanic dataset (sns.load_dataset("titanic")) containing around 900 passenger records, including features like age, gender, class, family size, fare, and port of embarkation, with the target variable indicating survival.

After dropping irrelevant features and analyzing missing values, I used `KNNImputer` to handle missing cabin data. I also engineered new features, such as family size and a combined "deck-class" variable, to better capture social stratification aboard the ship.

Stratified train-test splitting was used due to class imbalance (only 38% survived). I trained SVC, Logistic Regression, Random Forest, and Ridge Classifier using `RandomizedSearchCV`, optimizing for recall. The best parameter sets and performance metrics (F1, ROC AUC, train/test scores) were saved for further analysis.

SVC emerged as the best-performing model. I then used PCA to reduce dimensionality (2D and 3D) and visualized decision boundaries, confusion matrices, and ROC AUC scores for both train and test sets. I also built a `VotingClassifier` combining top models using both hard and soft voting, with soft voting giving the best results.

As an exploratory extension, I experimented with **unsupervised clustering**. Using a pipeline, I added `KMeans` cluster labels (with optimized `n_clusters`) as additional features for the SVC model. However, the added clusters did not improve test performance, possibly due to overfitting. I then tried **DBSCAN**, a density-based clustering algorithm, tuning the epsilon parameter using k-distance plots. With epsilon = 1.65, DBSCAN identified 6 clusters and 69 outliers, which were incorporated into the training data for further modeling.

### 🔍 Key Concepts:
- Feature engineering (deck-class combo, family size)
- Missing data imputation via KNN
- Stratified sampling for imbalanced targets
- Hyperparameter tuning with `RandomizedSearchCV`
- PCA-based decision boundary visualization
- Voting classifiers (hard & soft)
- Adding unsupervised cluster features via `KMeans` and `DBSCAN`

