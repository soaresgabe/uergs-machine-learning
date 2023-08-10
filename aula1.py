import numpy as np
import matplotlib.pyplot as plt

x = np.array([15,21,25,28,35,39,49])
y = np.array([5,7,9,8,9,11,15])

m = 20
b = 10

iter = 100000
alfa = 0.0001 # Learning Rate

y_predito = m * x + b
erro = y_predito - y

for i in range(iter):
    deriv_parcial_m = 2 * np.sum(x * erro)
    deriv_parcial_b = 2 * np.sum(erro)

    m = m - alfa * deriv_parcial_m
    b = b - alfa * deriv_parcial_b

    y_predito = m * x + b
    erro = y_predito - y 
    
    print(erro)

plt.scatter(x,y)
xreg = np.arange(0,50,1)
plt.plot(xreg,m*xreg+b,color="red")
plt.show()
