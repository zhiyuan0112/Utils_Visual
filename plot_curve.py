import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(400, 700, 31)
y = np.loadtxt('curve16')

for i in range(y.shape[0]):
    if i in [1,2,16,18]:
        plt.plot(x, y[i], 'w-', alpha=0)
    else:
        plt.plot(x, y[i])

plt.xlim([400,700])
plt.savefig('curve16.png')
plt.show()