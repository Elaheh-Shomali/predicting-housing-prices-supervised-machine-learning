# Predicting Housing Prices using Supervised Machine Learning

## Project Overview

This repository contains a machine learning project that predicts expensive houses based on a set of features. The project is divided into two major phases:

- **Classification Task**: Build a model to classify whether a house is expensive or not (“Expensive” or “Not expensive”).
- **Regression Task**: Build a model to predict the exact price of a house.

This division illustrates two key supervised learning tasks—classification and regression—and highlights their respective evaluation metrics.

## Methodology

### 1. Data Preprocessing

- Handled missing values and encoded categorical variables.
- Scaled numerical variables for standardization.

### 2. Feature Selection

- Conducted correlation analysis and assessed feature importance.

### 3. Model Development

- **Phase 1**: Implemented Decision Tree, K-Nearest Neighbors, Random Forest, and Logistic Regression for classification.
- **Phase 2**: Implemented advanced regression models, including Linear Regression, Decision Tree Regressor, and Stochastic Gradient Descent (SGD) Regressor, to enhance predictive accuracy.
- Tuned hyperparameters using GridSearchCV for optimal performance.

### 4. Model Evaluation and Key Metrics

- **Classification Metrics**: Accuracy, Confusion Matrix, Precision, recall, specificity, and F1-score.
- **Regression Metrics**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and R-squared.

## Tools and Techniques

- **Programming Language**: Python
- **Libraries**: pandas, NumPy, scikit-learn, matplotlib

## Key Learnings

- Data preprocessing improves model performance significantly.
- Cross-validation helps detect overfitting and underfitting.
- Hyperparameter tuning enhances model accuracy.
- Clear distinctions between classification and regression tasks aid in model evaluation.

## Challenges Overcome

- **Missing Data**: Implemented effective imputation strategies.
- **Model Selection**: Balanced simplicity and predictive power.
- **Task Differentiation**: Designed and evaluated models specific to classification and regression.

## Conclusion

This project showcases the power of supervised machine learning in predicting house prices through both classification and regression approaches. By applying rigorous preprocessing, cross-validation, and hyperparameter tuning, the models deliver reliable predictions. The techniques and insights from this project can be extended to other real-world problems.
