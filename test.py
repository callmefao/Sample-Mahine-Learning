import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('co2.csv')   # Đọc dữ liệu
data['co2'] = data['co2'].interpolate()     # Fill Nan
fig, ax = plt.subplots()
ax.plot(data['time'], data['co2'], label="Actual CO2")
plt.show()