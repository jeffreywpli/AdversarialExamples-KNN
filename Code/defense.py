
'''
This was originally designed to be flexible with what layer of the network was used for K-NN.
Initially, we used the normal 'hidden representation' (i.e., the last layer before the logits).
However, we found that using the logits was more effective.
As a result, this implementation could be made cleaner/more efficient.
'''

from keras.models import Model as KerasModel
import sys
import tensorflow as tf

sys.path.insert(0, "cleverhans")
from cleverhans.model import Model

from nearest_neighbors import knn

class DefendedModel(Model):

    '''
    Params:
    model -  either a MNISTModel or CIFARModel object
    X - The 'layer_name' representation of real and adversarial images
    Y - A value of -1 indicates a real image and 1 indicates an adversarial image
    K - the number of nearest neighbors to use
    '''
    def __init__(self, model = None, X = None, Y = None, K = 1, bias = 0.0):
        super(DefendedModel, self).__init__()

        if model is None:
            raise ValueError('model argument must be supplied.')

        # 10 image classes + adversarial class
        self.num_classes = 11

        # Get the Keras model from the input model class
        self.model = model.model

        self.X = X
        self.Y = Y
        self.K = K
        self.bias = bias

    def get_logits(self, x):
        with tf.name_scope("DefendedLogits") as scope:
            # Get the hidden represtnation and the logits from the classifier
            with tf.name_scope("VerboseBaseModel") as scope:
                logits_model = self.model(x)
            with tf.name_scope("KNN") as scope:
                votes = knn(logits_model, self.X, self.Y, self.K, bias = self.bias)
            with tf.name_scope("AdversarialLogit") as scope:
                logit_adv = tf.sign(votes) * 2 * tf.reduce_max(tf.abs(logits_model), reduction_indices = [1])
            return tf.concat([logits_model, tf.expand_dims(logit_adv, 1)], axis = 1)
