
'''
Warning:  make sure that the range on line 1683 of cleverhans/attacks_tf.py is set to [-0.5, 0.5]
'''
from keras import backend as Keras
from keras.backend.tensorflow_backend import set_session
import multiprocessing
import numpy as np
import os
import pickle
import sys
import tensorflow as tf

sys.path.insert(0, "../../Code/nn_breaking_detection")
from setup_mnist import MNISTModel, MNIST
from setup_cifar import CIFARModel, CIFAR

sys.path.insert(0, "../../Code/cleverhans")
from cleverhans.model import Model
from cleverhans.attacks import SPSA

sys.path.insert(0, "../../Code/")
from defense import DefendedModel
from analysis import analysis

def run(args, restrict = True):
    if restrict:
        # Restrict the visible GPUs to the one for this subprocess
        id = np.int(multiprocessing.current_process().name.split("-")[1])
        os.environ["CUDA_VISIBLE_DEVICES"]= str(id - 1)

    # Load Parameters
    dataset = args[0]
    epsilon = float(args[1])
    mode = args[2]
    K = int(args[3])
    bias = float(args[4])

    fname = dataset + "/" + str(epsilon) + "_" + mode + "_" + str(K) + "_" + str(bias)

    # Configure Keras/Tensorflow
    Keras.clear_session()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))

    sess = Keras.get_session()
    Keras.set_learning_phase(False)

    # Fix Random Seeds
    np.random.seed(1)
    tf.set_random_seed(1) #Having this before keras.clear_session() causes it it hang for some reason

    # Load Model/Data and setup SPSA placeholders
    N = 1000
    if dataset == "MNIST":
        # Base Model
        base_model = MNISTModel("../1-Models/MNIST")
        data = MNIST()
        x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        # SPSA
        shape_spsa = (1, 28, 28, 1)
        x_spsa = tf.placeholder(tf.float32, shape = shape_spsa)
    elif dataset == "CIFAR":
        # Base Model
        base_model = CIFARModel("../1-Models/CIFAR")
        data = CIFAR()
        x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        # SPSA
        shape_spsa = (1, 32, 32, 3)
        x_spsa = tf.placeholder(tf.float32, shape = shape_spsa)
    y_spsa = tf.placeholder(tf.int32)

    # Load the hidden representations of the real and adversarial examples from the training set
    x_train_real = np.squeeze(np.load("../3-Representation/" + dataset + "/train_" + mode + ".npy"))
    x_train_adv = np.squeeze(np.load("../3-Representation/" + dataset + "/train_adv_" + mode + ".npy"))

    n_train = x_train_real.shape[0]
    n_train_adv = x_train_adv.shape[0]
    x_train = np.float32(np.vstack((x_train_real, x_train_adv)))
    #print("Bounds ", np.max(np.abs(x_train)))
    y_train = np.float32(np.hstack((-1.0 * np.ones(n_train), np.ones(n_train_adv))))

    # Create the defended model
    model_defended = DefendedModel(base_model, x_train, y_train, K, bias = bias)
    defended_logits = model_defended.get_logits(x)

    # Get the predictions on the original images
    labels = np.argmax(data.test_labels[:N], axis = 1)
    logits_real = sess.run(defended_logits, {x: data.test_data[:N]})
    fp = (np.argmax(logits_real, axis = 1) == 10) #False positives of the defense
    pred_undefended = np.argmax(np.delete(logits_real, -1, axis=1), axis = 1) #Original model prediction

    # Configure the attack
    attack = SPSA(model_defended, back = "tf", sess = sess)
    with tf.name_scope("Attack") as scope:
        gen = attack.generate(x_spsa, y_target = y_spsa, epsilon = epsilon, is_targeted = True,
                                num_steps = 100, batch_size = 2048, early_stop_loss_threshold = -5.0)

    # Run the attack
    pred_adv = -1.0 * np.ones((N, 10))
    for i in range(N):
        if i % 10 == 0:
            print(fname, " ", i)
            out = {}
            out["FP"] = fp
            out["Labels"] = labels
            out["UndefendedPrediction"] = pred_undefended
            out["AdversarialPredictions"] = pred_adv
            file = open(fname, "wb")
            pickle.dump(out, file)
            file.close()

        x_real = data.test_data[i].reshape(shape_spsa)

        # Try a targeted attack for each class other than the original network prediction and the adversarial class
        for y in range(10):
            if y != pred_undefended[i]:
                x_adv = sess.run(gen, {x_spsa: x_real, y_spsa: y})
                pred_adv[i,y] = np.argmax(sess.run(defended_logits, {x: x_adv}))

    out = {}
    out["FP"] = fp
    out["Labels"] = labels
    out["UndefendedPrediction"] = pred_undefended
    out["AdversarialPredictions"] = pred_adv
    file = open(fname, "wb")
    pickle.dump(out, file)
    file.close()

    analysis(fname)

if __name__ == "__main__":
    args = [[sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]]]
    run(args[0], restrict = False)

