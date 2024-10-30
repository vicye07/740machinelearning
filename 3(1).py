import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
kesai = 0.1
def V(kesai,t):
    if abs(t)<=kesai:
        return t**2/2*kesai
    else:   
        return abs(t)-kesai/2
y = [V(kesai,t) for t in np.linspace(-1,1,100)]
plt.plot(np.linspace(-1,1,100),y)
plt.xlabel('t')
plt.ylabel('V(t)')
plt.title('V(t) as a function of t')
plt.show()