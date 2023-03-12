import pandas as pd
import numpy as np

housing = pd.read_csv("data.csv")

# print(housing.head())
# print(housing.info())
# print(housing.describe())

import matplotlib.pyplot as plt
# housing.hist(bins=50, figsize=(40, 40))
# plt.show()


# def split_train_test(data, test_ratio):
#     shuffled = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled[ : test_set_size]
#     train_indices = shuffled[test_set_size : ]
#     return data.iloc[train_indices], data.iloc[test_indices]

# train_set, test_set = split_train_test(housing, 0.2)
# print(f"rows in train set: {len(train_set)} \n rows in test set: {len(test_set)} \n ")

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# print(f"rows in train set: {len(train_set)} \n rows in test set: {len(test_set)} \n ")

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["CHAS"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# print(strat_test_set["CHAS"].value_counts())    
# print(strat_test_set["CHAS"].info())
# print(strat_test_set["CHAS"].describe())

# print(strat_train_set["CHAS"].value_counts())    
# print(strat_train_set["CHAS"].info())
# print(strat_train_set["CHAS"].describe())

corr_matrics = housing.corr()

# print(corr_matrics["MEDV"].sort_values(ascending=False))

from pandas.plotting import scatter_matrix

attributes = ["MEDV", "RM", "ZN", "LSTAT"]

# print(scatter_matrix(housing[attributes], figsize=(12,10)))
scatter_matrix(housing[attributes], figsize=(12,10))
# plt.show()

housing["TPRM"] = housing["TAX"] / housing["RM"]
# print(housing["TPRM"])

# corr_matrics = housing.corr()
# print(corr_matrics["MEDV"].sort_values(ascending=False))      

housing.plot(kind="scatter", x = "TPRM", y = "MEDV", alpha = 0.8)
# plt.show()

housing = strat_train_set.drop("MEDV", axis = 1)
housing_labels = strat_train_set["MEDV"].copy()

#  For Missing Values

# print(housing.info())

# Option-1
# a = housing.dropna(subset=["RM"])
# print(a.shape)

# Option-2
# b = housing.drop("RM", axis=1)
# print(b.shape)

# Option-3
# mean = housing["RM"].mean()
# print(mean)
# print(housing["RM"].fillna(mean))

median = housing["RM"].median()
# print(median)
housing["RM"].fillna(median)
# print(housing["RM"].fillna(median))

# mode = housing["RM"].mode()
# print(mode)
# print(housing["RM"].fillna(mode))


# Filling Missing Values Using Imputer

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)

# print(imputer.statistics_)

X = imputer.transform(housing)
housing_tr = pd.DataFrame(X, columns = housing.columns)
# print(housing_tr.describe())


# Scikit-learn Design

# Creating Pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
  ("imputer", SimpleImputer(strategy = "median")),
  ("std_scaler", StandardScaler())  
])

housing_num_tr = pipeline.fit_transform(housing_tr)
# print(housing_num_tr.shape)


# Selecting model for Dregon Real Estates Dataset

# Model-1 LinearRegressin()

from sklearn.linear_model import LinearRegression  
# model = LinearRegression()  

# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
# rmse_scores = np.sqrt(-scores)

# def print_scores(scores):
#   print("Scores: ", scores)
#   print("Mean: ", scores.mean())
#   print("Standard Deviation: ", scores.std())

# print_scores(rmse_scores)

# model.fit(housing_num_tr, housing_labels)

# some_data = housing.iloc[ : 5]
# some_label = housing_labels.iloc[ : 5]

# prepared_data = pipeline.transform(some_data)

# pre_data = model.predict(prepared_data)
# print(list(pre_data))
# print(list(some_label))

# from sklearn.metrics import mean_squared_error

# housing_predictions = model.predict(housing_num_tr)

# mse = mean_squared_error(housing_labels, housing_predictions)
# rmse = np.sqrt(mse)

# print(mse)
# print(rmse)

# Model Output

# Model -> LinearRegression
# Mean: 4.221894675406022
# Standard Deviation: 0.7520304927151625

# implementing LinearRegression() to find the mse and rmse of the model

# Model-2 DecisionTreeRegression

from sklearn.tree import DecisionTreeRegressor
# model = DecisionTreeRegressor()

# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
# rmse_scores = np.sqrt(-scores)

# def print_scores(scores):
#   print("Scores: ", scores)
#   print("Mean: ", scores.mean())
#   print("Standard Deviation: ", scores.std())

# print_scores(rmse_scores)

# model.fit(housing_num_tr, housing_labels)

# some_data = housing.iloc[ : 5]
# some_label = housing_labels.iloc[ : 5]

# prepared_data = pipeline.transform(some_data)

# pre_data = model.predict(prepared_data)
# print(list(pre_data))
# print(list(some_label))

# from sklearn.metrics import mean_squared_error

# housing_predictions = model.predict(housing_num_tr)

# mse = mean_squared_error(housing_labels, housing_predictions)
# rmse = np.sqrt(mse)

# print(mse)
# print(rmse)

# Model Output

# Model -> DecisionTreeRegressor
# Mean: 4.289504502474483
# Standard Deviation: 0.848096620323756

# Overfit the model 

# Model-3 RandomForestRegressor()

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

def print_scores(scores):
  print("Scores: ", scores)
  print("Mean: ", scores.mean())
  print("Standard Deviation: ", scores.std())
  
# print_scores(rmse_scores)

model.fit(housing_num_tr, housing_labels)

some_data = housing.iloc[ : 5]
some_label = housing_labels.iloc[ : 5]

prepared_data = pipeline.transform(some_data)

pre_data = model.predict(prepared_data)
# print(list(pre_data))
# print(list(some_label))

from sklearn.metrics import mean_squared_error

housing_predictions = model.predict(housing_num_tr)

mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)

# print(mse)
# print(rmse)

# Model Output

# Model -> RandomForestRegression
# Mean:  3.319517128493299
# Standard Deviation:  0.6189366410149104

# good to go with RandomTreeRegression

# Save and Load the machine learning model

from joblib import dump, load
dump(model, "Real_Estates.joblib")

# Testing the model on test data

X_test = strat_test_set.drop("MEDV", axis = 1)
Y_test = strat_test_set["MEDV"].copy()

X_test_prepared = pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)

final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

# print(final_rmse)
# print(final_predictions, list(Y_test))

# Using the model for predictions

# print(prepared_data[0])

features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747, -0.23746478,
 -1.31238772,  2.61111401, -1.0016859,  -0.5778192,  -0.97491834,  0.41164221,
 -0.86091034]])

result = model.predict(features)
# print(result)