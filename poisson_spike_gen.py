import numpy as np
import sklearn
import matplotlib.pyplot as plt
import coloredlogs, logging


logger = logging.getLogger("poisson_Gen")
coloredlogs.install(level='INFO')

""" Tester Poisson spike generator """

# seed for same results
seed = np.random.seed(7)
# parameters for simulation of spike train
firing_rate = 100      # Hz
dt = 1 / 1000          # seconds
total_interval = 10    # simulation for 10 ms


def plotter(spikemat, time_dur):
    for id in range(spikemat.shape[0]):
        spikepos = np.take(time_dur, np.where(spikemat[id, :] == 1))
        plt.plot(spikepos, id, '|', c='k')


    plt.ylim(top=spikemat.shape[0]+1)
    plt.plot([0]*20, [i for i in range(20)], c='blue')
    plt.xlabel('Time(ms)')
    plt.ylabel('Trial number')
    plt.show()

def poisson_spike_generator(fr, sim_len, n_trials):
    dt = 1 / 1000    # we need to convert it to seconds dt is taken to be 1 ms
    total_interval = int(np.floor(sim_len / dt))
    raw_data = np.random.rand(n_trials, total_interval)
    spike_mat = np.where(raw_data < fr * dt, 1, 0)
    end_val = sim_len - dt
    time_vec = np.linspace(0, end_val, num=total_interval)
    return spike_mat, time_vec


def homogenous_poisson_gen():
    """ firing rate constant over time """
    pass

def inhomogenous_poisson_gen():
    """ time dependent firing rate """
    pass

################# Baseline of 6 hz and after presentation of moving stimulus, firing rate of this neuron increases to 30 Hz ############
matdata, timedure = poisson_spike_generator(30, 1, 20)
basmat, bastime = poisson_spike_generator(6, 0.5, 20)
bastime = (bastime - bastime[-1]) * 1000 - 1

# concatenate columns wise
result_mat = np.concatenate((basmat, matdata), axis=1)
result_time = np.concatenate((bastime, timedure*1000))

plotter(result_mat, result_time)

##################
