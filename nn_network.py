import tensorflow as tf
import matplotlib.pyplot as plt
import timeit
from tensorflow.keras import layers, activations
import coloredlogs, logging
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

logger = logging.getLogger("nn_network")
coloredlogs.install(level='INFO')


#========================================================#
# Model PARAMETERS #
#========================================================#
seed = 5
my_init = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=seed)

""" FCN network as mentioned in the paper """

input = layers.Input(shape=(784,), name='input')
dense1 = layers.Dense(1200, use_bias=False, kernel_initializer=my_init, name='dense1')(input)
relu1 = activations.relu(dense1)
droput1 = layers.Dropout(rate=0.5, seed=seed)(relu1)
dense2 = layers.Dense(1200, use_bias=False, kernel_initializer=my_init, name='dense2')(droput1)
relu2 = activations.relu(dense2)
dropout2 = layers.Dropout(rate=0.5, seed=seed)(relu2)
dense3 = layers.Dense(10, use_bias=False, kernel_initializer=my_init, name='dense3')(dropout2)
out = activations.softmax(dense3)
model = tf.keras.Model(inputs=input, outputs=out, name='FCN_relu')
# just print out the model
tf.keras.utils.plot_model(model, to_file='FCN_relu.png', show_shapes=True)


#=================================================================================#
# Data setup #
#=================================================================================#
(train_images, train_targets), (test_images, test_targets) = tf.keras.datasets.mnist.load_data()
# data statsgtgt
print (f"[INFO] Training images found {train_images.shape[0]}")
print (f"[INFO] Test images found {test_images.shape[0]}")

# reshape images to the 784 input size
train_images = train_images.reshape(train_images.shape[0], 784).astype('float32') / 255
test_images = test_images.reshape(test_images.shape[0], 784).astype('float32') / 255

""" trying with k fold cross validation for model performance """
#train_X, val_X, train_y, val_y = train_test_split(train_images, train_targets, test_size=0.20, random_state=seed)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cv_scores = []
split_count = 0
for train, val in kfold.split(train_images, train_targets):
    # tensorboard callback
    logdir="logs/" + "split-" + str(split_count)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, profile_batch=3)
    # lets try training the model
    model.compile(optimizer= tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(),
             metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])
    # fit now
    training_scores = model.fit(train_images[train], train_targets[train], batch_size=100, epochs=50, verbose=0, callbacks=[tensorboard_callback])
    # evaluate the model
    val_scores = model.evaluate(train_images[val], train_targets[val], verbose=0, callbacks=[tensorboard_callback])
    print ("==================model metrics===========\n")
    print (f"[training stats] avg loss: {np.average(training_scores.history['loss']):5.2f} avg accuracy: {np.average(training_scores.history['sparse_categorical_accuracy']) * 100:5.2f}")
    print (f"[validation stats] loss: {val_scores[0]:5.2f}, accuracy: {val_scores[1] * 100:5.2f}")
    cv_scores.append(val_scores[1] * 100)
    split_count += 1
print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))

# evalute on test set
results = model.evaluate(test_images, test_targets, batch_size=100)
print (f"test loss {results[0]:5.2f} test acc: {results[1] * 100:5.2f}")

predictions = model.predict(test_images, batch_size=100)
y_preds = tf.argmax(predictions, axis=1)
print ("=========================\n")
print(classification_report(test_targets, y_preds))

# BUG: serialize model to JSON error in TF*2.0 while serialing model
#model.save('model.h5')
# serialize weights to HDF5
model.save_weights("model.h5")
print("model wts saved")
