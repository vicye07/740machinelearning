import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def Guassain_kernel(x,y,sigma):
    return np.exp(-np.linalg.norm(x-y)**2/(2*sigma**2))

def polynomial_kernel(x,y,h,sigma):
    return np.sum([np.dot(x,y)**i/sigma**(2*i-2) for i in range(1,h+1)])

def kernel_matrix(X,X_test,kernel,sigma):
    kernel_matrix = np.zeros((X.shape[0],Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X_test.shape[0]):
            kernel_matrix[i,j] = kernel(X[i],X_test[j],sigma)

def kernel_regression(X,Y,kernel,sigma,lamda):
    K = kernel_matrix(X,X,kernel,sigma)
    one_coloum = np.zeros((X.shape[0]))
    K_hat = np.hstack((one_coloum,K))
    K_prime = np.block([[np.ones(K.shape[0],K.shape[1]),np.ones(1,K.shape[0])],
                        [np.ones(K.shape[0],1),K]])
    alpha = np.dot(np.linalg.inv(np.dot(K_hat.t,K_hat)+lamda*K_prime),K_hat.T,Y)
    return alpha 

def kernel_predict(X,Y,X_test,kernel,sigma,lamda):
    alpha = kernel_regression(X,Y,kernel,sigma,lamda)
    alpha = alpha[1:]
    alpha0 = alpha[0]
    K = kernel_matrix(X,X_test,kernel,sigma)
    return np.dot(K.T,alpha)+alpha0

def residual(X,Y,X_test,Y_test,kernel,sigma,lamda):
    return np.sum(np.square(Y_test-kernel_predict(X,Y,X_test,kernel,sigma,lamda)))
    
data = pd.read_csv('project3_F2024_2_Train.csv').to_numpy()
X = data[0:10]
Y = data[10:]
data_test = pd.read_csv('project3_F2024_2_Test.csv').to_numpy()
X_test = data_test[0:10]
Y_test = data_test[10:]
sigma = 2.5
kernel = Guassain_kernel
lamda = np.linspace(0.001,0.1,100)
RSS_total = []
for i in lamda:
    RSS = residual(X,Y,X_test,Y_test,kernel,sigma,lamda)
    RSS_total.append(RSS)
plt.plot(lamda,RSS_total)
plt.xlabel('lamda')
plt.ylabel('RSS')
plt.title('RSS as a function of lamda')
plt.show()