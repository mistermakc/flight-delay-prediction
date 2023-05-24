# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             recall_score, precision_score, f1_score)
from sklearn.model_selection import KFold, ShuffleSplit, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# %%
# We lead the data.
df_19 = pd.read_csv('Jan_2019_ontime.csv')
df_20 = pd.read_csv('Jan_2020_ontime.csv')

# %% [markdown]
# # 1. Knowing the data

# %%
# First look at the data
df_19.head()

# %%
# Detailled information abou the data
df_19.info()

# %%
# We see the 2020 data.
df_20.head()

# %%
# We see the 2020 info.
df_20.info()

# %%
# Describing outcome variable in 2019
df_19['ARR_DEL15'].describe()

# %%
# Describing outcome variable in 2019
df_20['ARR_DEL15'].describe()

# %%
# Checking if the data sets have the same columns so we can later merge them.
print(list(df_20.columns) == list(df_19.columns))

# %% [markdown]
# # 2. Data preparation

# %%
# Checking all categorical data
df_19.select_dtypes(include=['object']).columns

# %%
# Checking all numerical data (int and float)
df_19.select_dtypes(include=['float64','int64']).columns

# %%
# Checking all categorical data
df_20.select_dtypes(include=['object']).columns

# %%
# Checking all numerical data (int and float)
df_20.select_dtypes(include=['float64','int64']).columns

# %% [markdown]
# ## Preparing the merger of both datasets

# %%
# Creating a column for each dataset to then concatenate both datasets
df_19['YEAR'] = 2019
df_20['YEAR'] = 2020

# Checking if 'YEAR' is in both datasets
if 'YEAR' in df_19.columns and 'YEAR' in df_20.columns:
    print(True)
else:
    print(False)

# %%
print('2019 dataset shape ' + str(df_19.shape))
print('2020 dataset shape ' + str(df_20.shape))

# Creating one dataset
data = pd.concat([df_19,df_20])
print('Unique dataset shape ' + str(data.shape))

# %%
data.info()

# %% [markdown]
# ## Concatenation of arrival and departure delay 
# Instead of doing multi classification, we will merge both clumns in one column (boolean). Notice that the data doesn't provide us with the exact amount of time delayed, so we only want to know if the flight was delayed or not.

# %%
data['DELAYED'] = (data['ARR_DEL15'].astype(bool) | data['DEP_DEL15'].astype(bool)).astype(int)
data.head()

# %% [markdown]
# ## Ploting the delay percentage per airline

# %%
# Replacing the carrier code by full airline names
cc = pd.read_csv('carrier_codes.csv', sep=";")

# Creating a dictionary from the first and third columns
rename_dict = dict(zip(cc.iloc[:, 0], cc.iloc[:, 2]))

# Replacing the values in the first column with the corresponding values from the dictionary
data["AIRLINE"] = data["OP_UNIQUE_CARRIER"].replace(rename_dict)

data["AIRLINE"].unique()

# %%
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['hatch.linewidth'] = 1.8

# Calculate the percentage of delayed flights per airline
df_delayed = data.groupby('AIRLINE').agg({'DELAYED': 'sum', 'YEAR': 'count'})
df_delayed['DELAY_PERCENTAGE'] = df_delayed['DELAYED'] / df_delayed['YEAR']

# Sort the DataFrame by percentage delay in descending order
df_delayed = df_delayed.sort_values('DELAY_PERCENTAGE', ascending=False)

fig = plt.figure(1, figsize=(15, 9))
ax = sns.barplot(x="YEAR", y=df_delayed.index, data=df_delayed, color="lightskyblue", errorbar=None)

# Use the percentage of delayed flights to set the hatch line
for i, (index, row) in enumerate(df_delayed.iterrows()):
    x = row['DELAY_PERCENTAGE']
    total_flights = row['YEAR']
    delayed_flights = row['DELAYED']
    
    # Draw the hatched bar using ax.barh()
    ax.barh(i, delayed_flights, color="lightskyblue", hatch='///', alpha=0.5, linewidth=1.8)

    # Add percentage text above the hatch area
    ax.text(delayed_flights, i, f"{x * 100:.1f}%", color='black', ha='left', va='center', fontsize=10)

ax.yaxis.label.set_visible(False)
plt.title('Percentage of Delayed Flights by Airline')
plt.xlabel('Total Number of Flights', fontsize=14, labelpad=10)

# Create custom legend handles and labels
handles = [Patch(facecolor='lightskyblue', edgecolor='lightskyblue', label='On time'),
           Patch(facecolor='black', hatch='///', linewidth=1.8, label='Flight delayed')]

plt.legend(handles=handles, loc='best')
plt.show()

# %%
# Showing the percentage of delayed flights including cancellations and divertions
average_delay_percentage = (data['DELAYED'].sum() / data['DELAYED'].count()).mean() * 100
print(f"Delayed, cancelled or diverted flights: {average_delay_percentage:.2f}%")

# %% [markdown]
# ## Dropping irrelevant columns for model prediction

# %%
# Getting rid of unnecessary columns
data.drop(['OP_CARRIER_AIRLINE_ID','TAIL_NUM','OP_CARRIER_FL_NUM','ORIGIN_AIRPORT_ID',
            'ORIGIN_AIRPORT_SEQ_ID','DEST_AIRPORT_ID','DEST_AIRPORT_SEQ_ID','Unnamed: 21',
            'OP_CARRIER', 'AIRLINE','ARR_DEL15','DEP_DEL15'], axis=1, inplace=True)

data.head()

# %% [markdown]
# ## Detecting null or empty values
# We can see that there are only null values in the DEP_TIME and ARR_TIME variables. This can be attributed to the cancelled or diverted flights since these flights either don't have a departure time (for cancelled flights) of arrival time.

# %%
print(data.isna().sum())

# %% [markdown]
# ## Splitting the problem
# 1. Predicting if a flight will be delayed or not.
# 2. Predicting if a flight will be cancelled or not.
# 3. Predicting if a flight will be diverted or not.
# 
# A flight delay, cancellation or divertion means a different cost for the company, so it makes sence to predict these cases separately. 

# %%
# Checking number of canceled flights
cancelled_flights_num = (data['CANCELLED'] == 1).sum()
print('Number of cancelled flights in dataset: ' + str(cancelled_flights_num))

# Checking number of diverted flights
diverted_flights_num = (data['DIVERTED'] == 1).sum()
print('Number of diverted flights in dataset: ' + str(diverted_flights_num))

# %% [markdown]
# ## A. Creating dataset for cancelled flights

# %%
cancelled_flights = data[data['CANCELLED'] == 1].copy()
cancelled_flights.drop(['DIVERTED'], axis=1, inplace=True)
cancelled_flights.head()

# %%
null_rows_cancelled = cancelled_flights[cancelled_flights.isna().any(axis=1)]
null_rows_cancelled.info()

# %% [markdown]
# ### Summary: arrival time is always null for the cancelled flights:

# %%
print(f'Number of rows with ARR_TIME null: {cancelled_flights["ARR_TIME"].isnull().sum()}')
print(f'Number of rows with DEP_TIME null: {cancelled_flights["DEP_TIME"].isnull().sum()}')

# %% [markdown]
# ## B. Creating dataset for diverted flights

# %%
diverted_flights = data[data['DIVERTED'] == 1].copy()
diverted_flights.drop(['CANCELLED'], axis=1, inplace=True)
diverted_flights.head()

# %% [markdown]
# ### Summary: there are only 482 null values and it's only for the variable ARR_TIME:

# %%
null_rows_diverted = diverted_flights[diverted_flights.isna().any(axis=1)]
null_rows_diverted.info()

# %% [markdown]
# -> As our main focus is on predicting flight delays, we won't continue pursuing cancellations or divertions for our main problem.

# %% [markdown]
# # 3. Data cleaning

# %%
# Working only with the data that doesn't have cancelled or diverted flights
print(cancelled_flights.isnull().sum())

# %%
# Working only with the data that doesn't have cancelled or diverted flights
df = data[(data['CANCELLED'] == 0) & (data['DIVERTED'] == 0)]
print(df.isnull().sum())

# %% [markdown]
# We can see that we don't have any null values without the cancelled and diverted flights.

# %%
df.head()

# %% [markdown]
# We want to drop the columns CANCELLED and DIVERTED because they are always 0.

# %%
# Dropping the
df = df.drop(columns=['CANCELLED', 'DIVERTED'])

# %%
# Checking the skewness of the data
print(df.agg(['skew']).transpose())

# %%
df.head()

# %%
df.info()

# %% [markdown]
# ## Data types

# %%
# Getting the object columns
object_columns =  list(df.dtypes[df.dtypes == 'object'].index)
print(f"The number of object columns is: {len(strings_columns)}")
    
# Getting int or float columns
numeric_columns = list(df.drop(strings_columns,axis=1))
print(f"The number of numeric columns is: {len(numeric_columns)}")


# %% [markdown]
# ## Duplicates

# %%
df.duplicated().sum()

# %%
# Dropping small number of duplicates
df.drop_duplicates(inplace=True)
df.duplicated().sum()

# %% [markdown]
# # 4. Feature selection and visualization

# %%
# Selecting only continuous columns
continuous_columns = ['DEP_TIME', 'ARR_TIME', 'DISTANCE']
continuous_df = df[continuous_columns]

fig, axes = plt.subplots(nrows=1, ncols=len(continuous_columns), figsize=(20, 6))

for i, column in enumerate(continuous_columns):
    axes[i].hist(continuous_df[column], bins=100, color='lightskyblue')
    axes[i].set_title(column)
    axes[i].grid()

plt.show()


# %%
# DAY_OF_MONTH
plt.figure(figsize=(15, 6))
sns.countplot(data=df, x='DAY_OF_MONTH', color='lightskyblue')
plt.title('Flights by Day of the Month')
plt.xlabel('Day of the Month')
plt.ylabel('Number of Flights')
plt.grid(axis='y')
plt.show()

# DAY_OF_WEEK
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='DAY_OF_WEEK', color='lightskyblue')
plt.title('Flights by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Flights')
plt.grid(axis='y')
plt.show()

# DEP_TIME
plt.figure(figsize=(15, 6))
plt.hist(df['DEP_TIME'], bins=24, color='lightskyblue')
plt.title('Departure Time Distribution')
plt.xlabel('Departure Time (hours)')
plt.ylabel('Number of Flights')
plt.grid()
plt.show()

# ARR_TIME
plt.figure(figsize=(15, 6))
plt.hist(df['ARR_TIME'], bins=24, color='lightskyblue')
plt.title('Arrival Time Distribution')
plt.xlabel('Arrival Time (hours)')
plt.ylabel('Number of Flights')
plt.grid()
plt.show()

# DELAYED
plt.figure(figsize=(6, 6))
sns.countplot(data=df, x='DELAYED', color='lightskyblue')
plt.title('Delayed vs. On-time Flights')
plt.xlabel('Flight Status')
plt.ylabel('Number of Flights')
plt.grid(axis='y')
plt.xticks([0, 1], ['On-time', 'Delayed'])
plt.show()


# %% [markdown]
# ## Checking percentage of delayed flights

# %%
print(f"Delayed flights: {round((num_delayed/len(df))*100,2)}%")

# %% [markdown]
# ## Checking outliers

# %%
# Selecting only relevant columns for outliers
outliers_columns = ['DAY_OF_WEEK', 'DAY_OF_MONTH', 'DEP_TIME', 'ARR_TIME', 'DISTANCE']
outliers_df = df[outliers_columns]

def check_outliers(outliers_df):
    for col in outliers_columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.grid()
        sns.boxplot(x=outliers_df[col], color='lightskyblue', whis=1.5)  # Adjust the whis parameter
        plt.show()

check_outliers(outliers_df)

# %% [markdown]
# The only variable that shows indications of outliers is "DISTANCE". However we won't handle this scenario since it is possible that a person may have taken a flight between two distant countries, resulting in a significant distance that would be considered an outlier in our data. Therefore, we won't change the distance data.

# %% [markdown]
# ## Checking variable correlation

# %%
fig,ax = plt.subplots(figsize=(15,9))
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.show()

# %% [markdown]
# As we can see, the variables don't have much correlation except for arrival time and departure time which makes sence.

# %% [markdown]
# ## Encoding the "YEAR" variable

# %%
df['YEAR'] = pd.factorize(df['YEAR'])[0]
df['is_2020'] = (df['YEAR'] == 1).astype(int)
df['is_2019'] = (df['YEAR'] == 0).astype(int)

# Delete the original "year" column
df.drop('YEAR', axis=1, inplace=True)

df.head()

# %%
df.info()

# %% [markdown]
# ## Encoding all object variables

# %%
def encode_categories(features):
    lb_make = LabelEncoder()
    for i in range(len(features)):
        df[features[i]] = lb_make.fit_transform(df[features[i]])

# %%
encode_categories(['OP_UNIQUE_CARRIER', 'ORIGIN' , 'DEST' , 'DEP_TIME_BLK'])

# %%
df.info()

# %% [markdown]
# ## Full correlation map

# %%
fig,ax = plt.subplots(figsize=(15,9))
sns.heatmap(df.corr(),annot=True,cmap="Blues")
plt.show()

# %% [markdown]
# Due to the high correlation between the DEP_TIME_BLK and DEP_TIME variable, we dropped the DEP_TIME_BLK variable. However, the cross validation and model scores improve when including it in the dataset. Hence, it is logical to include it.

# %%
# Dropping the variable 
#df = df.drop(columns=['DEP_TIME_BLK'])

# %% [markdown]
# ## Balancing dataset

# %%
# Divide by class
count_ontime, count_delayed = df.DELAYED.value_counts()
ontime = df[df.DELAYED == 0]
delayed = df[df.DELAYED == 1]
test = pd.concat([ontime, delayed], axis=0)

# Check balance of dataset
print('Balance of dataset:')
print(test.DELAYED.value_counts())


# %%
# Undersampling the dataset with sklearn.resample based on exited users
ontime_under = ontime.sample(count_delayed)  # Sample count_delayed instances from the ontime class
bln = pd.concat([ontime_under, delayed], axis=0)

print('Random under-sampling:')
print(bln.DELAYED.value_counts())
print()

# %% [markdown]
# # 5. Modeling and predicting the data

# %%
# Dropping target variable
X = df.drop('DELAYED',axis=1)
y = df['DELAYED']

# %%
X.head()

# %%
y.head()

# %% [markdown]
# ## Splitting the data into train and test data

# %%
# Splitting the train and test data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=42)
cv = KFold(n_splits=5, random_state=1, shuffle=True)

# %%
print(f'Shape X train: {x_train.shape}')
print(f'Shape X test: {x_test.shape}')
print(f'Shape y train: {y_train.shape}')
print(f'Shape y test: {y_test.shape}')

# %% [markdown]
# ## Scaling the data

# %%
# Scaling the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# %%
def evaluate_model(y_true, y_pred, model_name, cv_scores=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    print(f"---- {model_name} ----")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{report}\n")

    if cv_scores is not None:
        cv_accuracy = cv_scores.mean()
        print(f"Cross-Validation Accuracy: {cv_accuracy:.4f}")


# %% [markdown]
# ### A. Logistic Regression

# %%
# Defining model
logreg = LogisticRegression(random_state=42, max_iter=3000)

#Cross validation
logreg_scores = cross_val_score(logreg, X, y, cv=cv, scoring='roc_auc')

# Fitting and predicting model
logreg.fit(x_train, y_train)
y_pred_logreg = logreg.predict(x_test)

# Model evaluation
evaluate_model(y_test, y_pred_logreg, "Logistic Regression", cv_scores=logreg_scores)

# %% [markdown]
# ### B. Decision Tree

# %%
# Defining model
dtree = DecisionTreeClassifier(random_state=42)

#Cross validation
dtree_scores = cross_val_score(dtree, X, y, cv=cv, scoring='roc_auc')

# Fitting and predicting model
dtree.fit(x_train, y_train)
y_pred_dtree = dtree.predict(x_test)

# Model evaluation
evaluate_model(y_test, y_pred_dtree, "Decision Tree", cv_scores=dtree_scores)

# %% [markdown]
# ### C. Random Forest

# %%
# Tuning hyperparameters
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Defining model and using Grid Search
rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(x_train, y_train)

# Providing best hyperparameters
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_
print("Best parameters:", best_params)

# Prediciting model with best estimators
y_pred = best_estimator.predict(x_test)
print(classification_report(y_test, y_pred))

# %%
# Defining model
randomforest = RandomForestClassifier(random_state=42)

#Cross validation
randomforest_scores = cross_val_score(randomforest, X, y, cv=cv, scoring='roc_auc')

# Fitting and predicting model

randomforest.fit(x_train, y_train)
y_pred_randomforest = randomforest.predict(x_test)

# Model evaluation
evaluate_model(y_test, y_pred_randomforest, "Random Forest", cv_scores=randomforest_scores)

# %% [markdown]
# ### D. Gradient Boosting

# %%
# Defining model
gradientboost = GradientBoostingClassifier(random_state=42)

#Cross validation
gradientboost_scores = cross_val_score(gradientboost, X, y, cv=cv, scoring='roc_auc')

# Fitting and predicting model
gradientboost.fit(x_train, y_train)
y_pred_gradientboost = gradientboost.predict(x_test)

# Model evaluation
evaluate_model(y_test, y_pred_gradientboost, "Gradient Boosting", cv_scores=gradientboost_scores)

# %% [markdown]
# ### E. K-Nearest Neighbor

# %%
# Defining model
knn = KNeighborsClassifier()

#Cross validation
knn_scores = cross_val_score(knn, X, y, cv=cv)

# Fitting and predicting model
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)

# Model evaluation
evaluate_model(y_test, y_pred_logreg, "Logistic Regression", cv_scores=knn_scores)

# %% [markdown]
# ### F. AdaBoost

# %%
# Defining model
adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=42), random_state=42)

# Cross validation
adaboost_scores = cross_val_score(adaboost, X, y, cv=cv, scoring='roc_auc')

# Fitting and predicting model
adaboost.fit(x_train, y_train)
y_pred_adaboost = adaboost.predict(x_test)

# Model evaluation
evaluate_model(y_test, y_pred_adaboost, "AdaBoost", cv_scores=adaboost_scores)

# %% [markdown]
# ### G. XGBoost

# %%
# Defining model
xgboost = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Cross validation
xgboost_scores = cross_val_score(xgboost, X, y, cv=cv, scoring='roc_auc')

# Fitting and predicting model
xgboost.fit(x_train, y_train)
y_pred_xgboost = xgboost.predict(x_test)

# Model evaluation
evaluate_model(y_test, y_pred_xgboost, "XGBoost", cv_scores=xgboost_scores)

# %% [markdown]
# # 6. Interpretation
# 
# **AdaBoost:**
# 
# Accuracy: 0.8892
# Precision: 0.7223
# Recall: 0.7064
# F1-score: 0.7143
# 
# The AdaBoost model achieved an accuracy of 0.8892, indicating that it correctly predicted the flight delay status for 88.92% of the instances. The precision of 0.7223 suggests that when the model predicts a flight to be delayed, it is correct 72.23% of the time. The recall of 0.7064 indicates that the model identified 70.64% of the actual delayed flights. The F1-score of 0.7143 represents the balance between precision and recall.
# 
# **Logistic Regression:**
# 
# Accuracy: 0.8468
# Precision: 0.9961
# Recall: 0.2196
# F1-score: 0.3598
# 
# The Logistic Regression model achieved an accuracy of 0.8468. However, it should be noted that the precision is 0.9961, which means that when the model predicts a flight to be delayed, it is correct 99.61% of the time. However, the recall is low at 0.2196, indicating that the model only identified 21.96% of the actual delayed flights. The F1-score is 0.3598, indicating the trade-off between precision and recall.
# 
# **XGBoost:**
# 
# Accuracy: 0.9020
# Precision: 0.9218
# Recall: 0.5464
# F1-score: 0.6861
# 
# The XGBoost model achieved an accuracy of 0.9020, indicating that it correctly predicted the flight delay status for 90.20% of the instances. The precision of 0.9218 suggests that when the model predicts a flight to be delayed, it is correct 92.18% of the time. The recall of 0.5464 indicates that the model identified 54.64% of the actual delayed flights. The F1-score of 0.6861 represents the balance between precision and recall.
# 
# **Gradient Boosting:**
# 
# Accuracy: 0.8755
# Precision: 0.9588
# Recall: 0.3813
# F1-score: 0.5456
# 
# The Gradient Boosting model achieved an accuracy of 0.8755, indicating that it correctly predicted the flight delay status for 87.55% of the instances. The precision of 0.9588 suggests that when the model predicts a flight to be delayed, it is correct 95.88% of the time. The recall of 0.3813 indicates that the model identified 38.13% of the actual delayed flights. The F1-score of 0.5456 represents the balance between precision and recall.
# 
# **Random Forest:**
# 
# Accuracy: 0.9091
# Precision: 0.9357
# Recall: 0.5760
# F1-score: 0.7130
# 
# The Random Forest model achieved an accuracy of 0.9091, indicating that it correctly predicted the flight delay status for 90.91% of the instances. The precision of 0.9357 suggests that when the model predicts a flight to be delayed, it is correct 93.57% of the time. The recall of 0.5760 indicates that the model identified 57.60% of the actual delayed flights. The F1-score of 0.7130 represents the balance between precision and recall.
# 
# **Decision Tree:**
# 
# Accuracy: 0.8894
# Precision: 0.7232
# Recall: 0.7059
# F1-score: 0.7144
# 
# The Decision Tree model achieved an accuracy of 0.8894, indicating that it correctly predicted the flight delay status for 88.94% of the instances. The precision of 0.7232 suggests that when the model predicts a flight to be delayed, it is correct 72.32% of the time. The recall of 0.7059 indicates that the model identified 70.59% of the actual delayed flights. The F1-score of 0.7144 represents the balance between precision and recall.

# %% [markdown]
# ## Final Recommendation
# 
# Based on the performance metrics of the different models, it is recommended to consider either the Random Forest or XGBoost model for predicting flight delays.
# 
# Both Random Forest and XGBoost exhibit relatively high accuracy, precision, recall, and F1-score compared to the other models. However, the Random Forest model slightly outperforms XGBoost in terms of recall, indicating a better ability to identify actual delayed flights.
# 
# Additionally, the Random Forest model has been optimized with the following parameters: criterion='gini', max_depth=None, min_samples_leaf=1, min_samples_split=2, and n_estimators=200. These hyperparameters have been tuned to improve the model's performance.
# 
# Therefore, the final recommendation is to use the Random Forest model with the provided optimized parameters for predicting flight delays.


