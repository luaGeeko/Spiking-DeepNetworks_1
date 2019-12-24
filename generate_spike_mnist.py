import numpy as np
from skimage.io import imread
from sklearn import datasets
import matplotlib.pyplot as plt
import gzip
import os


def extract_to_numpy(root_path):
    """ converts the ubyte.gz zipped format of mnist data to numpy format
    parameters:
    -----------------
    root_path(str) : path to the root directory of download mnist data

    returns:
    -----------------
    train_images: numpy array (N, 28 * 28)
    train_labels: numpy array (N, )
    test_images: numpy array (N, 28 * 28)
    test_labels: numpy array (N, )
    """

    def process(filepath, N, type='images'):
        with gzip.open(filepath) as bs:
            if type == 'images':
                bs.read(16)
                temp_buf = bs.read(28 * 28 * N)
                temp_data = np.frombuffer(temp_buf, dtype=np.uint8).astype(np.float32)
                temp_data = temp_data.reshape(N, 28*28)
                return temp_data

            elif type == 'labels':
                bs.read(8)
                temp_buf = bs.read(1 * N)
                temp_labels = np.frombuffer(temp_buf, dtype=np.uint8).astype(np.int64)
                return temp_labels

            else:
                pass


    train_images = process(os.path.join(root_path, 'train-images-idx3-ubyte.gz'), N=60000)
    train_labels = process(os.path.join(root_path, 'train-labels-idx1-ubyte.gz'), N=60000, type='labels')
    test_images = process(os.path.join(root_path, 't10k-images-idx3-ubyte.gz'), N=10000)
    test_labels = process(os.path.join(root_path, 't10k-labels-idx1-ubyte.gz'), N=10000, type='labels')

    return train_images, train_labels, test_images, test_labels

def test_generate_poisson_spike_data(image_arr, vis=True):
    """ test function to generate poisson distributed spike trains for a given image. the image has been normalized to values between 0 and 1
    default values have been taken for the following:
     1. input rate(max_rate) = 1000 Hz,
     2. time interval(dt) = 1 ms
     3. simulation time (sim_len) = 200 ms
    parameters:
    ----------------
    image_arr: image in numpy array format
    vis(boolean) : whether to plot the image
    returns:
    ---------------
    image: numpy array created from spikes
    """
    # lets work with one image first say i take the max rate = 200 Hz means 200 spikes/sec
    max_rate = 1000 # 1000 hz is the rate, which is 1000 spikes/second
    dt = 0.001 # delta T in s
    sim_len = 200 / 1000 # total simulation length (s)
    total_bins = int(np.floor(sim_len / dt))
    image = image_arr / 255.
    # probability of a single spike happening in the given dt = max_rate * dt
    new_image = np.zeros((image.shape[0],), dtype=np.float32)
    for i in range(total_bins):
        rescale_factor = 1 / (dt * max_rate)
        spike_snapshot = np.random.randn(image.shape[0],) * rescale_factor
        input_image = spike_snapshot <= image
        new_image = new_image + input_image.astype(np.float32)

    if vis:
        res = new_image.reshape(28, 28)
        plt.imshow(res, cmap='gray')
        plt.show()

    return new_image

################################ DUMMY USAGE CHECK #############################################
train_images, train_labels, test_images, test_labels = extract_to_numpy('/home/lua/Spiking-DeepNetworks_1/mnist_data')
spiked_image = test_generate_poisson_spike_data(train_images[1], vis=True)
