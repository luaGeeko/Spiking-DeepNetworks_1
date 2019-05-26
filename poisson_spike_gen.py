import numpy as np
import sklearn

""" Tester Poisson spike generator """

# seed for same results
seed = np.random.seed(7)
# parameters for simulation of spike train
firing_rate = 100      # Hz
dt = 1 / 1000          # seconds
total_interval = 10    # simulation for 10 ms

# generate random numbers uniform distribution [0, 1]
x = np.random.uniform(0, 1, total_interval)
poisson_spike_train = np.where(x < firing_rate * dt, 1, 0)
print (poisson_spike_train)
