
from keras import backend as Keras
from keras.backend.tensorflow_backend import set_session
import numpy as np
import sys
import tensorflow as tf

sys.path.insert(0, "../../Code/nn_breaking_detection/")
from setup_cifar import CIFARModel, CIFAR
from setup_mnist import MNISTModel, MNIST

sys.path.insert(0, "../../Code/cleverhans")
from cleverhans.model import Model

sys.path.insert(0, "../../Code/")
from defense import DefendedModel

dataset = sys.argv[1]
layer_name = sys.argv[2]
mode = sys.argv[3]
K = int(sys.argv[4])
bias = float(sys.argv[5])

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

sess = Keras.get_session()
Keras.set_learning_phase(False)

np.random.seed(1)
tf.set_random_seed(1)

if dataset == "MNIST":
    data = MNIST()
    model = MNISTModel("../1-Models/MNIST")
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
elif dataset == "CIFAR":
    data = CIFAR()
    model = CIFARModel("../1-Models/CIFAR")
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))

training_accuracy = np.mean(np.argmax(model.model.predict(data.train_data), axis = 1) == np.argmax(data.train_labels, axis = 1))
print("Training Accuracy: " + str(training_accuracy))
testing_accuracy = np.mean(np.argmax(model.model.predict(data.test_data), axis = 1) == np.argmax(data.test_labels, axis = 1))
print("Testing Accuracy: " + str(testing_accuracy))

X = data.train_data
X_adv = np.load("../2-AEs/" + dataset + "/train_" + mode + ".npy")

pred_original = model.model.predict(X)
pred_adv = model.model.predict(X_adv)
print("Adversarial Success Rate: " + str(1 - np.mean(np.argmax(pred_original) == np.argmax(pred_adv))))

delta = X - X_adv

l_1 = np.sum(np.abs(delta), axis = (1,2,3))
print("Mean l1 distortion: " + str(np.mean(l_1)))
print("Max l1 distortion: " + str(np.max(l_1)))

l_2 = np.sqrt(np.sum(delta ** 2, axis = (1,2,3)))
print("Mean l2 distortion: " + str(np.mean(l_2)))
print("Max l2 distortion: " + str(np.max(l_2)))

l_i = np.max(np.abs(delta), axis = (1,2,3))
print("Mean l_inf distortion: " + str(np.mean(l_i)))
print("Max l_inf distortion: " + str(np.max(l_i)))

print("Defense Analysis on first 1000 testing images")
x_train_real = np.squeeze(np.load("../3-Representation/" + dataset + "/train_" + mode + ".npy"))
x_train_adv = np.squeeze(np.load("../3-Representation/" + dataset + "/train_adv_" + mode + ".npy"))

n_train = x_train_real.shape[0]
n_train_adv = x_train_adv.shape[0]
x_train = np.float32(np.vstack((x_train_real, x_train_adv)))
y_train = np.float32(np.hstack((-1.0 * np.ones(n_train), np.ones(n_train_adv))))

model_defended = DefendedModel(model, layer_name, x_train, y_train, K, bias = bias)
defended_logits = model_defended.get_logits(x)

logits_real = sess.run(defended_logits, {x: data.test_data[:1000]})

fpr = (np.argmax(logits_real, axis = 1) == 10)
err = (np.argmax(logits_real, axis = 1) != np.argmax(data.test_labels[:1000], axis = 1))

pred_undefended = np.argmax(np.delete(logits_real, -1, axis=1), axis = 1)
err_u = (pred_undefended != np.argmax(data.test_labels[:1000], axis = 1))

print("Defense FPR: ", np.mean(fpr))
print("Defense Error Rate: ", np.mean(err))
print("Undefended Error Rate: ", np.mean(err_u))
