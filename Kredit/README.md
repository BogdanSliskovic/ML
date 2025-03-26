## Credit Risk Modeling (Binary Classification)

This project uses the Credit dataset from OpenML, containing over 17,000 records labeled as "good" or "bad" credit risk. I experimented with three classification models: Logistic Regression, Support Vector Machines, and Random Forest — each with a tailored preprocessing strategy.

For Logistic Regression, I applied standardization and one-hot encoding with the first category dropped to avoid multicollinearity. For SVM, I used full one-hot encoding without dropping the first column. Random Forest was trained on raw (unprocessed) features due to its robustness to feature scaling and encoding.

The primary focus of this project was to understand how models behave in terms of overfitting and underfitting. I visualized F1-scores on both training and test sets while varying the training size and key hyperparameters. I also used grid search to optimize hyperparameters for all models, using scoring='recall' to prioritize reducing false negatives.

Random Forest showed the best performance on the test set, so I performed a deeper analysis using:
- Feature importance with confidence intervals
- SHAP values for individual prediction interpretation
- ROC curves across different tree depths
- Analysis of misclassified test samples

### Key Concepts:
- Custom preprocessing per algorithm
- Model selection using recall as the scoring metric
- Detecting and avoiding overfitting/underfitting via learning curves
- Model interpretation via SHAP and feature importance

