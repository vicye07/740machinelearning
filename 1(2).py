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
r = 10
d = 3
D = np.diag([3,3,3])
beta = solve_sylvester(np.dot(X.T,X),r*D,np.dot(X.T,Y))
a0 = beta[0].T
b = beta[1:]
print(b.shape)
j_values = np.arange(Y.shape[1])  

plt.figure(figsize=(10, 6))
plt.scatter(j_values, a0, label='a0(j)')
plt.scatter(j_values, b[0, :], label='b(1, j)')
plt.scatter(j_values, b[1, :], label='b(2, j)')
plt.scatter(j_values, b[2, :], label='b(3, j)')

plt.title('Values of a0(j), b(1, j), b(2, j), and b(3, j) as functions of j')
plt.xlabel('j')
plt.ylabel('Values')
plt.legend()
plt.grid(True)


plt.show()