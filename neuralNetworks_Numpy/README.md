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


# 📄 Report: Wage Estimation and Sampling Analysis Based on LSMS Serbia 2007

---

## 1. Introduction

This report analyzes data from the **Living Standards Measurement Study (LSMS)** conducted in Serbia in 2007, focusing on estimating the **mean and total monthly wage**, and exploring the determinants of wage differences among the population.

Main goals:
- Determine how education, age, region, gender, and working hours affect wage.
- Evaluate sampling methods: Simple Random (PSU) vs Stratified Sampling (SSU).
- Estimate GDP based on income data.

---

## 2. Data Overview

**Variables used**:

- `plata`: Net wage from main job (previous month)
- `obrazovanje`: Years of education (based on ISCED)
- `obr3`: Education categories (basic, secondary, higher)
- `starost`: Age in years
- `satiRada`: Weekly working hours
- `zene`: Gender (Male/Female)
- `urban`: Urban vs Rural
- `region`: Four regions (Beograd, Vojvodina, Zapadna + Šumadija, Jugoistočna Srbija)

---

## 3. Data Cleaning

Steps taken:
- Removed missing values (especially in `satiRada`)
- Cast `float` variables to `int` to optimize memory
- Excluded retired population: males > 65, females > 60
- Outliers removed via:
  - **Interquartile range (IQR)** on `satiRada`
  - Derived `satnica = plata / (satiRada × 52 / 12)`
  - Removed all values below legal minimum hourly wage (55 RSD)

**Result**: From 5,141 observations → cleaned to 3,199 (~38% removed)

---

## 4. Normality Testing & Bootstrap

- JB test rejected normality of wage.
- Created `Bootstrap` function to resample wage means.
- 10,000 samples of size 319 (10% of dataset) drawn with replacement.
- Mean wage distribution passed JB test → approximated normal.

### Confidence Intervals

Used quantiles to compute 90%, 95%, 99% confidence intervals.

### Sample Size Analysis

Derived required sample sizes for each confidence level:
- Sample sizes for 90%, 95%, 99% are very similar.
- Final sample size chosen: **296**

---

## 5. Regression Analysis

- Model: `wage ~ education + working_hours + age + gender + region + urban`
- R² ≈ 0.265 (26.5%)
- All predictors statistically significant
- Residuals non-normal → tried **log-linear model**
  - Improved AIC/BIC
  - Coefficient on education ≈ 6.5% increase per year of schooling

---

## 6. Sampling Designs and Estimators

### 6.1 Simple Random Sampling (PSU)

- Sample size: 294 and 588
- Compared ratio vs regression estimation
- For smaller samples, **ratio estimator** had lower MSE
- Larger samples favored **regression estimator**

### 6.2 Stratified Sampling (SSU)

Stratified by:
- Region
- Gender
- Education

For each:
- Evaluated estimator bias, standard deviation, and MSE
- Found region-based stratification most precise
- Stratification by education yielded high variance due to grouped coding

---

## 7. GDP Estimation

Using weighted average wage and population distributions (from census 2011):
- Estimated **total income**
- Added taxes and gross operating surplus
- **Estimated GDP close to official 2007 value**

---

## 8. Conclusion

This report demonstrates:
- Proper application of statistical inference techniques
- Bootstrap-based validation of distributional assumptions
- Custom implementation of OLS and estimators
- Practical use of sampling theory for public policy modeling

---

