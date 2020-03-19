import numpy as np
import matplotlib.pyplot as plt

result = np.loadtxt('testXp.txt', delimiter = '\t', skiprows = 1)

result = result[result[:,3].argsort()]
np.set_printoptions(suppress=True)
print(result[-10:])

# IoU vs every other parameter
iou = result[:,3]
lr = result[:,2]
#plt.plot(lr, iou)
#plt.show()
print(result[:,1])