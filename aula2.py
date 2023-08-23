import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x,y = make_regression(n_samples=500, n_features=1, noise=30, random_state=5)
plt.scatter(x,y)

model=LinearRegression()
model.fit(x,y)

b_reg = model.intercept_ # coef linear
m_reg = model.coef_ # coef angular

x_reg = np.arange(-4,4,1)

plt.scatter(x,y)
y_reg = m_reg * x_reg + b_reg
plt.plot(x_reg,y_reg,'red')
#plt.show()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

model.fit(x_train,y_train)
result = model.score(x_test, y_test)
print(result)
