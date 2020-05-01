import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("./data/中国_cov.csv")
data = data[['date', 'confirmed']]
data['date'] = pd.to_datetime(data['date'])
data.index = data['date']
data = data.sort_index()
# total = data[1].cumsum()
total = data[['confirmed']]
total = total.reset_index()['confirmed']
# print(total)

total.index = total.index + 1
plt.scatter(total.index, total)
plt.show()

linear_reg = LinearRegression()
x_data = total.index[:, np.newaxis]
y_data = total[:, np.newaxis]
linear_reg.fit(x_data, y_data)

# plt.scatter(x_data, y_data)
plt.plot(x_data, linear_reg.predict(x_data))
# plt.show()

plt.scatter(x_data, y_data, linewidths=1)
poly = PolynomialFeatures(9)
x_data_poly = poly.fit_transform(x_data)
linear_reg = LinearRegression()
linear_reg.fit(x_data_poly, y_data)
plt.plot(x_data, linear_reg.predict(x_data_poly), color='red', linewidth=2.0, linestyle='--')
# x_data_poly = [[i ** 0, i ** 1, i ** 2, i ** 3, i ** 4, i ** 5, i ** 6, i ** 7, i ** 8, i ** 9] for i in
#                np.arange(1, 30)]
# plt.scatter(np.arange(1, 30), linear_reg.predict(x_data_poly))

plt.legend(["Linear predict data", "Poly predict data", "Real data"])
plt.xlabel("Daily update")
plt.ylabel("CoVID Confirmed")
plt.savefig('covid.png')
plt.show()

