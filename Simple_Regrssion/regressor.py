import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ydata_profiling import ProfileReport
from lazypredict.Supervised import LazyRegressor

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


data = pd.read_csv("StudentScore.xls", delimiter=",")
# profile = ProfileReport(data, title="Score Report", explorative=True)
# profile.to_file("score.html")

target = "writing score"

x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(missing_values=-1, strategy="median")),
    ("scaler", StandardScaler())
])

education_values = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree",
                    "master's degree"]
gender_values = ["male", "female"]
lunch_values = x_train["lunch"].unique()
test_values = x_train["test preparation course"].unique()
ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[education_values, gender_values, lunch_values, test_values]))
])

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num_feature", num_transformer, ["reading score", "math score"]),
    ("ord_feature", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nom_feature", nom_transformer, ["race/ethnicity"]),
])



# reg = Pipeline(steps=[
#     ("preprocessor", preprocessor),
#     ("model", HuberRegressor())
# ])

# Lazypredict
# reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = reg.fit(x_train, x_test, y_train, y_test)
# print(models)

#                                Adjusted R-Squared  R-Squared  RMSE  Time Taken
# Model
# HuberRegressor                               0.94       0.94  3.84        0.03
# LassoCV                                      0.94       0.94  3.85        0.06
# ElasticNetCV                                 0.94       0.94  3.86        0.07
# BayesianRidge                                0.94       0.94  3.86        0.06
# Ridge                                        0.94       0.94  3.86        0.01
# RidgeCV                                      0.94       0.94  3.86        0.02

# #GridSearch
# param_grid = {
#     'model__epsilon': [1.05, 1.35, 1.5],
#     'model__alpha': [0.0001, 0.001, 0.01],
#     'model__max_iter': [100, 200, 300],
#     'model__tol': [1e-05, 1e-04, 1e-03]
# }
#
# # Set up GridSearchCV
# grid_search = GridSearchCV(reg, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
#
# # Fit the model
# grid_search.fit(x_train, y_train)
#
# # Print best parameters
# print("Best parameters found: ", grid_search.best_params_)
#
# # Get the best model from the grid search
# best_model = grid_search.best_estimator_


# Final Code:
reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", HuberRegressor(alpha=0.001, epsilon=1.5, max_iter=100, tol=1e-05))
])

reg = reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)

# Calculate regression metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Regression Report:")
print(f"R²: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Regression Report:
# R²: 0.9382
# Mean Squared Error (MSE): 14.8977
# Mean Absolute Error (MAE): 3.1982
# Root Mean Squared Error (RMSE): 3.8598
