# Breast Cancer Classification using KNN

Early detection of breast cancer can make a life-saving difference. In this project, I built a complete machine learning pipeline to classify tumors as malignant or benign using the Wisconsin Diagnostic Breast Cancer dataset. The goal was not just to apply KNN, but to do it the right way, with proper preprocessing, validation, and model selection, the way it would be done in a real-world ML workflow.

This project focuses on how distance-based models behave on high-dimensional medical data and how critical steps like feature scaling and hyperparameter tuning directly impact performance.

## Problem Statement

Given numerical features extracted from cell nuclei images, the task is to predict whether a tumor is:

* Malignant (M)
* Benign (B)

This is a binary classification problem where both accuracy and reliability matter, since misclassification in medical domains has serious consequences.

## Dataset

* Source: Breast Cancer Wisconsin (Diagnostic)
* Total samples: 569
* Features: 30 continuous variables describing cell characteristics
* Target: Diagnosis (Malignant / Benign)
* The dataset is clean and well-structured, but feature ranges vary a lot, which makes scaling essential for KNN.

## Approach

### 1. Data Preprocessing

* Removed non-informative ID column
* Encoded the target labels
* Applied StandardScaler to normalize all features

Scaling is a crucial step here because KNN relies entirely on distance. Without normalization, features with larger numeric ranges would dominate the model and distort the neighborhood structure.

### 2. Model Building

I used K-Nearest Neighbors as a baseline yet powerful non-parametric classifier. It is simple in concept but very sensitive to data representation, which makes it a good choice to demonstrate the importance of preprocessing and tuning.

### 3. Hyperparameter Tuning

Instead of choosing K arbitrarily, I evaluated multiple values of K and tracked validation accuracy to find the point where the model generalizes best.

This helped balance:

* Overfitting at very small K
* Oversmoothing at very large K

The final model was selected based on test performance, not training accuracy.

## Results

The tuned KNN model achieves strong accuracy in distinguishing malignant from benign tumors, showing that even classical algorithms can perform very well when engineered carefully.

More importantly, the project demonstrates:

* Proper ML pipeline design
* The effect of feature scaling on distance-based models
* Systematic hyperparameter selection

## Why this project matters

Most KNN projects stop at “fit and predict.” This one focuses on:

* Clean preprocessing
* Reproducible experimentation
* Data-driven hyperparameter selection
* Framing the problem from a real diagnostic perspective

It reflects how such a model would actually be built in a healthcare analytics or applied ML environment.

## Tech Stack

* Python
* NumPy, Pandas
* Scikit-learn
* Matplotlib / Seaborn (for analysis and visualization)

## Future Improvements

* Compare KNN with SVM, Random Forest, and Gradient Boosting
* Add ROC-AUC, precision-recall, and confusion matrix analysis
* Apply PCA to visualize class separation
* Deploy the trained model as a simple REST API using FastAPI

---


