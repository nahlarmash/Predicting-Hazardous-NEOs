<h1 align="center">
Predicting Hazardous NEOs (Nearest Earth Objects)
<h1 align="center">
<img width="600" alt="image" src="https://github.com/nahlarmash/Predicting-Hazardous-NEOs/blob/main/Nearest%20Earth%20Objects.png">
</h1> 

<a name="readme-top"></a>
## Table of Contents
- [Overview](#Overview)
- [Data Importing and Cleaning](#Data-Importing-and-Cleaning)
- [Dataset Description](#Dataset-Description)
- [Exploratory Data Analysis (EDA)](#Exploratory-Data-Analysis-(EDA))
- [Data Preprocessing](#Data-Preprocessing)
- [Model Building and Evaluation](#Model-Building-and-Evaluation)
- [Conclusion](#Conclusion)

## Overview
In this project, we analyze and predict whether Near-Earth Objects (NEOs), which have been monitored by NASA from 1910 to 2024, are classified as hazardous. This is a binary classification problem where the goal is to predict if a given NEO is hazardous based on its features.

The dataset contains 338,199 records, each representing a celestial object, and includes attributes such as the object's diameter, velocity, miss distance, and more. Our task is to train several machine learning models and evaluate their ability to predict hazardous NEOs.


## Data Importing and Cleaning
In this section, we started by importing necessary libraries, reading the dataset, and cleaning the data.

- **Libraries Imported:**
  - `pandas`, `numpy`: For data manipulation and analysis.
  - `matplotlib.pyplot`, `seaborn`: For data visualization.
  - `sickit-learn`: For building machine learning models.
  - `imblearn`: To handle class imbalance.
  - `scikit-learn.metrics`: For evaluation metrics.

- **Data Importing:**
  The dataset was read into a pandas DataFrame.

- **Data Cleaning:**
  - Handled missing values by filling them with median.
     

## Dataset Description
The dataset used for this project includes records of NEOs with various features such as:
- **Absolute Magnitude:** The brightness of the object.
- **Estimated Diameter (min, max):** The estimated range of the NEO's diameter.
- **Relative Velocity:** The Speed of NEO relative to Earth.
- **Miss Distance:** The closest distance at which the NEO will pass by Earth.
- **Is Hazardous:** The target variable, where True indicates the NEO is hazardous, and False means it is not.

## Exploratory Data Analysis (EDA)
We performed EDA to understand the data distribution and relationships between features.

1. **Plotting Distributions:**
  - Histograms were created for features like `absolute_magnitude`, `estimated_diameter`, `relative_velocity`, and `miss_distance` to understand their distribution.
  - A bar plot was used to compare the count of hazardous vs non-hazardous NEOs.

- **Key Insights from Plotting Distributions:**
  - Most NEOs have moderate brightness.
  - The majority of NEOs are small, but a few have larger diameters.
  - NEOs typically travel at moderate speeds, though there are some high-speed outliers.

2. **Correlation Analysis:**
  - A heatmap was used to visually represent the strength and direction of correlations between features.

- **Key Insights From Correlation Heatmap:**    
  - `absolute_magnitude` had a moderate inverse correlation with `estimated_diameter`, indicating that brighter objects tend to have smaller.
  - `relative_velocity` and `miss_distance` have a weakly correlations with other features.


## Data Preprocessing
This section involves preparing the data for model training.

- **Feature Selection:**
  Dropped columns `neo_id`, `name`, `orbiting_body` because they don't provide predictive value.

- **Label Encoding:**
  Applied label encoding to transform the binary target variable (`is_hazardous`) into numerical form.

- **Feature Scaling:**
  Used `StandardScaler` to scale numerical features.

- **Handling Class Imbalance:**
  The dataset was imbalanced, with significantly fewer hazardous NEOs. We used `SMOTE`(Synthetic Minority Over-sampling Technique) to generate synthetic samples for the minority class.


## Model Building and Evaluation
Several machine learning models were trained to classify hazardous NEOs:

- **Model Building:**
1. **Logistic Regression:**
   - A linear model that predicts the probability of a binary class.
   - Trained using the `LogisticRegression` class from `scikit-learn`.
  
2. **Decision Tree Classifier:**
   - A tree-based model where each node represents a decision based on a feature.
   - Trained Using `DecisionTreeClassifier`.

3. **Random Forest Classifier:**
   - An ensemble model that combines the predictions of multiple decision trees to improve performance.
   - Trained using `RandomForestClassifier`.

- **Model Evaluation:**
  After training the models, we evaluated their performance using several metrics:
  - **Accuracy:** The proportion of correct predictions.
  - **Precision:** The proportion of positive predictions that are actually positive.
  - **Recall:** The proportion of actual positives that are correctly identified.
  - **F1-Score:** The harmonic mean of precision and recall, useful for imbalanced datasets.
  - **AUC-ROC:** Measures the ability of the model to distinguish between classes.
 
1. **Logistic Regression Evaluation:**
  - **Accuracy:** 79.52%
  - **Precision:** 75.25%
  - **Recall:** 87.93%
  - **F1-Score:** 81.10%
  - **AUC-ROC:** 0.7953

2. **Decision Tree Evaluation:**
  - **Accuracy:** 91.69%
  - **Precision:** 91.18%
  - **Recall:** 92.30%
  - **F1-Score:** 91.73%
  - **AUC-ROC:** 0.9169
    
3. **Random Forest Evaluation:**
  - **Accuracy:** 94.92%
  - **Precision:** 93.79%
  - **Recall:** 96.20%
  - **F1-Score:** 94.98%
  - **AUC-ROC:** 0.9492

> **Random Forest performed best across all metrics, making it the most suitable model for this task.**


## Conclusion
In this notebook, we successfully trained machine learning models to predict hazardous NEOs. The **Random Forest Classifier** provided the best performance, with high accuracy, precision, recall, F1-score, and AUC-ROC score, making it a reliable model for identifying potentially dangerous celestial objects.
