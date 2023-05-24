# Flight Delay Prediction

This project aims to predict flight delays using machine learning algorithms. The dataset used in this project consists of flight data from January 2019 and January 2020. The objective is to build a model that can accurately predict whether a flight will be delayed or not.

## Dataset

The dataset used in this project includes various features such as departure time, arrival time, distance, airline, and more. It contains both numerical and categorical variables. The dataset was split into train and test sets for model training and evaluation.

## Data Preparation

Before training the models, the dataset underwent several data preparation steps. This included checking for missing values, encoding categorical variables, scaling numerical variables, and handling outliers. The dataset was also balanced to address any class imbalance issues.

## Models and Evaluation

Several machine learning models were trained and evaluated on the dataset. The models used in this project include Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, K-Nearest Neighbors, AdaBoost, and XGBoost. Performance metrics such as accuracy, precision, recall, and F1-score were used to evaluate the models.

## Results

Based on the evaluation, the Random Forest and XGBoost models performed the best among all the models tested. Both models exhibited high accuracy, precision, recall, and F1-score. However, the Random Forest model slightly outperformed the XGBoost model in terms of recall. Therefore, the Random Forest model with optimized parameters was recommended as the preferred model for predicting flight delays.

## Conclusion

In conclusion, this project demonstrated the effectiveness of machine learning algorithms in predicting flight delays. The Random Forest model, in particular, showed promising results and can be utilized for real-world applications to assist in decision-making related to flight operations and planning.

Please refer to the provided Jupyter Notebook or Python scripts for more detailed code implementation and analysis.

