
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import numpy as np
import sys
import tensorflow as tf

sys.path.insert(0, "../../Code/")
from nearest_neighbors import knn

# Fix Random Seed
np.random.seed(1)
tf.set_random_seed(1)

# Load Parameters
dataset = sys.argv[1]
mode = sys.argv[2]
bias = np.float(sys.argv[3])

fname = dataset + "/" + mode + "_" + str(bias)

# Load the representations of the real and adversarial images used to train K-NN
x_train_real = np.squeeze(np.load("../3-Representation/" + dataset + "/train_" + mode + ".npy"))
x_train_adv = np.squeeze(np.load("../3-Representation/" + dataset + "/train_adv_" + mode + ".npy"))

x_val_real = np.squeeze(np.load("../3-Representation/" + dataset + "/val_" + mode + ".npy"))
x_val_adv = np.squeeze(np.load("../3-Representation/" + dataset + "/val_adv_" + mode + ".npy"))

# Merge that data and label it
n_train = x_train_real.shape[0]
n_train_adv = x_train_adv.shape[0]
x_train = np.vstack((x_train_real, x_train_adv))
y_train = np.hstack((-1.0 * np.ones(n_train), np.ones(n_train_adv)))

n_val = x_val_real.shape[0]
n_val_adv = x_val_adv.shape[0]
x_val = np.vstack((x_val_real, x_val_adv))
y_val = np.hstack((-1.0 * np.ones(n_val), np.ones(n_val_adv)))

# Configure Keras/Tensorflow
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

K.set_learning_phase(False)

# Configure K-NN
if dataset == "MNIST":
    dim = 10
elif dataset == "CIFAR":
    dim = 10
X = tf.placeholder(tf.float32, shape=(None, dim))
X_test = tf.placeholder(tf.float32, shape=(None, dim))
Y = tf.placeholder(tf.float32, shape=(None))
K = tf.placeholder(tf.int32)

defense = knn(X_test, X, Y, K, bias = bias)

sample = np.random.choice(x_val.shape[0], 1000, replace = False)
x = x_val[sample]
y = y_val[sample]

indices_real = np.where(y == -1)[0]
indices_adv = np.where(y == 1)[0]

# Run K-NN
file = open(fname + ".txt", "w")
for k in [1,3,5,11,21,51,71]:
    
    votes = sess.run(defense, {X_test: x, X: x_train, Y: y_train, K: k})
    
    acc = np.mean(np.sign(votes) == y)
    fpr = np.mean(np.sign(votes[indices_real]) == 1)
    tpr = np.mean(np.sign(votes[indices_adv]) == 1)
    
    file.write(str(k) + "-NN" + "\n")
    file.write("Accuracy: " + str(acc) + "\n")
    file.write("FPR: " + str(fpr) + "\n")
    file.write("TPR: " + str(tpr) + "\n")

file.close()
