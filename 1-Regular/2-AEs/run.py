'''
Warning:  make sure that the range on line 1683 of cleverhans/attacks_tf.py is set to [-0.5, 0.5]
'''

from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import numpy as np
import sys
import tensorflow as tf

sys.path.insert(0, "../../Code/nn_breaking_detection")
from setup_mnist import MNISTModel, MNIST
from setup_cifar import CIFARModel, CIFAR

sys.path.insert(0, "../../Code/cleverhans")
from cleverhans.model import Model
from cleverhans.attacks import CarliniWagnerL2, MadryEtAl

sys.path.insert(0, "../../Code/")
from wrapper import Wrapper

# Fix Random Seed
np.random.seed(1)
tf.set_random_seed(1)

# Load Parameters
dataset = sys.argv[1]
mode = sys.argv[2]

# Configure Keras/Tensorflow
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

K.set_learning_phase(False)

# Configure Attack
if mode == "1":
    confidence = 0
elif mode == "2":
    confidence = 2
elif mode == "3":
    ord = np.inf
elif mode == "4":
    ord = 2
elif mode == "5":
    ord = 1

if dataset == "MNIST":
    base_model = MNISTModel("../1-Models/MNIST")
    data = MNIST()
    
    # Only ever tested mode=1
    if mode == "1" or mode == "2":
        learning_rate = 0.1
        binary_search_steps = 5
        max_iterations = 2000
        initial_const = 1.0
    elif mode == "3" or mode == "4" or mode == "5":
        epsilon = 0.1
    
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    shape = (0, 28, 28, 1)

elif dataset == "CIFAR":
    base_model = CIFARModel("../1-Models/CIFAR")
    data = CIFAR()
    
    if mode == "1" or mode == "2":
        learning_rate = 0.01
        binary_search_steps = 3
        max_iterations = 200
        initial_const = 0.01
    # For mode=4,5, these epsilon values where chosen based off the observered characteristics of the C&W attack
    # For mode=3, the defense was not robust and, for mode=4,5, K-NN could not separate the examples
    elif mode == "3":
        epsilon = 8/255
    elif mode == "4":
        epsilon = 0.7
    elif mode == "5":
        epsilon = 25

    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    shape = (0, 32, 32, 3)

batch_size = 1000

wrap_model = Wrapper(base_model)
if mode == "1" or mode == "2":
    attack = CarliniWagnerL2(wrap_model, back = "tf", sess = sess)
    gen = attack.generate(x, confidence = confidence, batch_size = batch_size, learning_rate = learning_rate, binary_search_steps = binary_search_steps, max_iterations = max_iterations, abort_early = True, initial_const = initial_const, clip_min = -0.5, clip_max = 0.5)
if mode == "3" or mode == "4" or mode == "5":
    attack = MadryEtAl(wrap_model, back = "tf", sess = sess)
    gen = attack.generate(x, eps = epsilon, ord = ord, clip_min = -0.5, clip_max = 0.5)

# Run Attack
X = data.train_data
X_adv = np.zeros(shape)
for i in range(0, X.shape[0], batch_size):
    x_adv = sess.run(gen, {x: X[i:i+batch_size]})
    X_adv = np.concatenate((X_adv, x_adv))
    print("train ", i)
np.save(dataset + "/train_" + mode, X_adv)

X = data.validation_data
X_adv = np.zeros(shape)
for i in range(0, X.shape[0], batch_size):
    x_adv = sess.run(gen, {x: X[i:i+batch_size]})
    X_adv = np.concatenate((X_adv, x_adv))
    print("val ", i)
np.save(dataset + "/val_" + mode, X_adv)
