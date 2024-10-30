import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def gaussian_kernel(x, y, sigma):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

def polynomial_kernel(x, y, h, sigma):
    return np.sum([np.dot(x, y) ** i / sigma ** (2 * i - 2) for i in range(1, h + 1)])

def compute_kernel_matrix(X, X_test, kernel, sigma):
    if kernel == gaussian_kernel:
        dists = cdist(X, X_test, 'sqeuclidean')
        return np.exp(-dists / (2 * sigma ** 2))
    else:
        # Assume polynomial kernel
        K = np.zeros((X.shape[0], X_test.shape[0]))
        for i in range(1, h + 1):
            K += (np.dot(X, X_test.T) ** i) / sigma ** (2 * i - 2)
        return K

def kernel_regression_model(X, Y, X_test, Y_test, kernel, sigma, lamda):
    K = compute_kernel_matrix(X, X, kernel, sigma)
    one_column = np.ones((X.shape[0], 1))
    K_hat = np.hstack((one_column, K))
    K_prime = np.block([[np.zeros((1, 1)), one_column.T], [one_column, K]])
    
    Y = Y.reshape(-1, 1)
    alpha = np.linalg.inv(K_hat.T @ K_hat + lamda * K_prime) @ K_hat.T @ Y
    
    alpha0 = alpha[0, 0]
    alpha_rest = alpha[1:]
    
    K_test = compute_kernel_matrix(X, X_test, kernel, sigma)
    Y_pred = K_test.T @ alpha_rest + alpha0
    residual = np.sum(np.square(Y_test - Y_pred.flatten()))
    
    return residual

# Load data
data = pd.read_csv('project3_F2024_2_Train.csv').to_numpy()
X = data[:, 0:10]
Y = data[:, 10:]
data_test = pd.read_csv('project3_F2024_2_Test.csv').to_numpy()
X_test = data_test[:, 0:10]
Y_test = data_test[:, 10:]

# Parameters
sigma = 2.5
kernel = gaussian_kernel
lamda_values = np.linspace(0.001, 0.1, 100)

# Compute residuals for each lambda
RSS_total = [kernel_regression_model(X, Y, X_test, Y_test, kernel, sigma, l) for l in lamda_values]

# Plotting
plt.plot(lamda_values, RSS_total)
plt.xlabel('lambda')
plt.ylabel('RSS')
plt.title('RSS as a function of lambda')
plt.show()
