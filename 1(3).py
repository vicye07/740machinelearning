import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve_sylvester
data = pd.read_csv('project3_F2024_1.csv').to_numpy()
Y = data[:,0:3]
X = data[:,3:]
X = pd.DataFrame(X)
X.insert(0, 'intercept', 1)
X = X.to_numpy()
r = 1000
d = 3
q = 200
D = np.zeros((q,q))
for i in range(q):
    if i == 0 or i ==q-1:
        D[i,i] = 1
    else:
        D[i,i] = 2
for i in range(q):
    for j in range(q):
        if j != i:
            D[i,j] = -1
beta = solve_sylvester(np.dot(X.T,X),r*D,np.dot(X.T,Y))
a0 = beta[0].T
b = beta[1:]
j_values = np.arange(Y.shape[0])  

plt.figure(figsize=(10, 6))
plt.plot(j_values, a0, label='a0(j)')
plt.plot(j_values, b[0, :], label='b(1, j)')
plt.plot(j_values, b[1, :], label='b(2, j)')
plt.plot(j_values, b[2, :], label='b(3, j)')

plt.title('Values of a0(j), b(1, j), b(2, j), and b(3, j) as functions of j')
plt.xlabel('j')
plt.ylabel('Values')
plt.legend()
plt.grid(True)


plt.show()