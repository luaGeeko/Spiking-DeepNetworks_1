import numpy as np
from scipy.io import loadmat
import coloredlogs, logging
import matplotlib.pyplot as plt


logger = logging.getLogger("tuning")
coloredlogs.install(level='INFO')

""" Tuning Curves Plotting """

# load mat file
data = loadmat("tuning.mat")
tune = data['tuningMat']          # (8, 360)
direction = data['direction'][0]
for id in range(tune.shape[0]):
    plt.plot(direction, tune[id, :], label='neuron '+ str(id))

plt.ylim(top=60)
plt.xlim(right=400)
plt.xlabel('Direction of motion')
plt.ylabel('Firing rate')
plt.legend(loc='best')
plt.show()
