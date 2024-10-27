import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from lazypredict.Supervised import LazyRegressor


# Tạo Training Dataset
def create_ts_data(data, window_size=5):
    i = 1
    while i < window_size:
        data[f"co2_{i}"] = data['co2'].shift(-i)
        i += 1
    data["co2_target"] = data['co2'].shift(-5)
    data = data.dropna(axis=0)
    return data


data = pd.read_csv('co2.csv')   # Đọc dữ liệu
data['time'] = pd.to_datetime(data['time'])     # Chuyển sang kiểu datetime
data['co2'] = data['co2'].interpolate()     # Fill Nan

data = create_ts_data(data)

x = data.drop(['time', 'co2_target'], axis=1)
y = data['co2_target']

train_ratio = 0.8
num_sample = len(x)

x_train = x[:int(num_sample * train_ratio)]
y_train = y[:int(num_sample * train_ratio)]
x_test = x[int(num_sample * train_ratio):]
y_test = y[int(num_sample * train_ratio):]

# reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = reg.fit(x_train, x_test, y_train, y_test)
# print(models)
huber = HuberRegressor()
huber.fit(x_train, y_train)
# huber_predict = huber.predict(x_test)
#
# tftr = TransformedTargetRegressor()
# tftr.fit(x_train, y_train)
# tftr_predict = tftr.predict(x_test)
#
linear = LinearRegression()
linear.fit(x_train, y_train)


# linear_predict = linear.predict(x_test)
#
# df = pd.DataFrame({
#     'Huber Predict': huber_predict,
#     'TFTR Predict': tftr_predict,
#     'Linear Predict': linear_predict,
#     'Actual CO2': y_test
# })
#
# fig, ax = plt.subplots()
# ax.plot(data[:int(num_sample * train_ratio)]['time'], data['co2'][:int(num_sample * train_ratio)], label="Actual CO2")
# ax.plot(data[int(num_sample * train_ratio):]['time'], data['co2'][int(num_sample * train_ratio):], label="Actual CO2")
# Vẽ y_predict
# ax.plot(data[int(num_sample * train_ratio):]['time'], huber_predict, label="huber_predict", linestyle='-')
# ax.plot(data[int(num_sample * train_ratio):]['time'], tftr_predict, label="tftr_predict ", linestyle='--')
# ax.plot(data[int(num_sample * train_ratio):]['time'], linear_predict, label="linear_predict", linestyle='-.')
# ax.set_xlabel("Year")
# ax.set_ylabel("CO2")
# ax.legend()
# ax.grid()
# plt.show()

# Bảng top các model tốt nhất (từ lazy predict)
#                                Adjusted R-Squared  R-Squared   RMSE  Time Taken
# Model
# HuberRegressor                               0.99       0.99   0.47        0.03
# TransformedTargetRegressor                   0.99       0.99   0.47        0.01
# LinearRegression                             0.99       0.99   0.47        0.01
# RANSACRegressor                              0.99       0.99   0.47        0.01
# LassoLarsCV                                  0.99       0.99   0.47        0.01


# Hàm tạo prediction dataframe
def create_prediction_df(matrix, model, prediction_number=10):
    for _ in range(prediction_number):
        current_row = matrix[-1]
        new_value = model.predict([current_row])[0]  # Predict based on the current row
        new_row = current_row[1:] + [new_value]  # Shift the row and add the new prediction
        matrix.append(new_row)
    return pd.DataFrame(matrix)


# Ví dụ: sử dụng với model LinearRegression và HuberRegressor
mtrx = [[380.5, 390, 390.2, 390.4, 393]]
linear_df = create_prediction_df(mtrx, linear, 10)
huber_df = create_prediction_df(mtrx, huber, 10)

print(linear_df)
# fig, ax = plt.subplots()
# ax.plot(data[:int(num_sample * train_ratio)]['time'], data['co2'][:int(num_sample * train_ratio)], label="Actual CO2")
# ax.plot(data[int(num_sample * train_ratio):]['time'], data['co2'][int(num_sample * train_ratio):], label="Actual CO2")
# Vẽ y_predict
# ax.plot(data[int(num_sample * train_ratio):]['time'], huber_predict, label="huber_predict", linestyle='-')
# ax.plot(data[int(num_sample * train_ratio):]['time'], tftr_predict, label="tftr_predict ", linestyle='--')
# ax.plot(data[int(num_sample * train_ratio):]['time'], linear_predict, label="linear_predict", linestyle='-.')
# ax.set_xlabel("Year")
# ax.set_ylabel("CO2")
# ax.legend()
# ax.grid()
# plt.show()
