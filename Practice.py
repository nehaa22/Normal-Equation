import pandas as pd
import numpy as np

reader = pd.read_csv("data.csv")
rows = 4
x = reader.iloc[:, 0].values
y = reader.iloc[:, 1].values

new_x = np.ones((rows, 1))
x = np.reshape(x, (rows, 1))
x = np.append(new_x, x, axis=1)
x_transpose = np.transpose(x)

x_transpose_dot_x = x_transpose.dot(x)

temp1 = np.linalg.inv(x_transpose_dot_x)
print(temp1)

temp2 = x_transpose.dot(y)
print(temp2)

theta = temp1.dot(temp2)
print(theta)