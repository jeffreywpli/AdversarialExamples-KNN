
import keras
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.models import Model as KModel
import numpy as np
import sys
import tensorflow as tf

sys.path.insert(0, "../../Code/nn_breaking_detection/")
from setup_cifar import CIFARModel, CIFAR
from setup_mnist import MNISTModel, MNIST

# Load Parameters
dataset = sys.argv[1]
layer_name = sys.argv[2]
mode = sys.argv[3]

# Configure Keras/Tensorflow
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

K.set_learning_phase(False)

# Load Model/Data
if dataset == "MNIST":
    data = MNIST()
    model = MNISTModel("../1-Models/MNIST").model
elif dataset == "CIFAR":
    data = CIFAR()
    model = CIFARModel("../1-Models/CIFAR").model

#print(model.summary()) #used to find what 'layer_name' should be
representation_layer = KModel(inputs = model.input, outputs=model.get_layer(layer_name).output)

# Compute Representation
np.save(dataset + "/train_" + mode, representation_layer.predict(data.train_data))
np.save(dataset + "/val_" + mode, representation_layer.predict(data.validation_data))

X_adv = np.load("../2-AEs/" + dataset + "/train_" + mode + ".npy")
np.save(dataset + "/train_adv_" + mode, representation_layer.predict(X_adv))

X_adv = np.load("../2-AEs/" + dataset + "/val_" + mode + ".npy")
np.save(dataset + "/val_adv_" + mode, representation_layer.predict(X_adv))
