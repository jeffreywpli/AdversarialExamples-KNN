
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
from defense_projection import DefendedModel

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

    fname = dataset + "/" + str(epsilon) + "_" + mode + "_" + str(K)

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
    N = 500
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
    y_train = np.float32(np.hstack((-1.0 * np.ones(n_train), np.ones(n_train_adv))))

    # Create the defended model
    defense = DefendedModel(base_model, x_train, y_train, K)
    get_votes = defense.get_votes(x)                   # Should this be get_votes, introducing separate method
    get_logits = defense.get_logits(x)

    # Configure the attack
    attack = SPSA(defense, back = "tf", sess = sess)
    with tf.name_scope("Attack") as scope:
        gen = attack.generate(x_spsa, y = y_spsa, epsilon = 0.01, is_targeted = False,
                                num_steps = 100, batch_size = 2048, early_stop_loss_threshold = -0.05)

    # Run the test
    sample = np.random.choice(data.test_data.shape[0], N, replace = False)
    x_sample = data.test_data[sample]
    y_sample = np.argmax(data.test_labels[sample], axis = 1)

    votes = sess.run(get_votes, {x: x_sample})

    count = 0
    bound = 0
    correct = 0
    for i in range(N):
        if votes[i,0] > 0:
            count += 1
            # Project via an adversarially attack on the votest
            #x_real = x_sample[i].reshape(shape_spsa)
            #x_adv = sess.run(gen, {x_spsa: x_real, y_spsa: 0}) #TODO: not adv, is projected
            x_proj = sess.run(get_logits, {x: x_sample[i]})
            projection_labels = np.argmax(x_proj, axis = 1)
            successful_projections = projection_labels[np.nonzero(projection_labels * (projection_labels != 10))]

            # Check if the projection was a success
            if successful_projections.shape[0] != 0:
                bound += 1

            # Check if the projection is predicted correctly
            if y_sample[i] == np.argmax(sess.run(get_logits, {x: x_proj}), axis = 1)[0]:
                correct += 1

    print("FP Count: ", count)
    print("FP Recovery in Bounds: ", bound / count)
    print("FP Recovery Accuracy: ", correct/count)


if __name__ == "__main__":
    args = [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]]
    run(args, restrict = False)

