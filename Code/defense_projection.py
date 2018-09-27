
import sys
import tensorflow as tf

sys.path.insert(0, "cleverhans")
from cleverhans.model import Model
from cleverhans.attacks import SPSA

from nearest_neighbors import knn

class KNN(Model):
    def __init__(self, model = None, X = None, Y = None, K = 1):
        super(KNN, self).__init__()

        self.num_classes = 2 # Class 0 is the adversarial vote, class 1 has a logit that is always zero
        self.model = model.model
        self.X = X
        self.Y = Y
        self.K = K

    def get_logits(self, x):
        with tf.name_scope("KNNLogits") as scope:
            logits = self.model(x)
            votes = knn(logits, self.X, self.Y, self.K)
            paddings = tf.constant([[0,0],[0,1]])
            return tf.pad(tf.expand_dims(votes, 1), paddings)

class DefendedModel(Model):
 
    '''
    Params:
    model -  either a MNISTModel or CIFARModel object
    X - The logit layer representation of real and adversarial images
    Y - A value of -1 indicates a real image and 1 indicates an adversarial image
    K - the number of nearest neighbors to use
    '''
    def __init__(self, model = None, X = None, Y = None, K = 1):
        super(DefendedModel, self).__init__()
        
        self.knn = KNN(model, X, Y, K)
        self.num_classes = 2 # TODO: 10
        self.model = model.model
    
    #TODO: get_votes
    def get_logits(self, x):
        with tf.name_scope("Votes") as scope:
            votes = self.knn.get_logits(x)
            return votes
            #return tf.squeeze(tf.slice(votes, [0, 0], [-1, 1]))
    
    # TODO: get_logits
    # TODO:  if K-NN says input is Adversarial, project to a nearby input that K-NN says is real and use that output
    def get_logits_real(self, x):
        with tf.name_scope("DefendedLogits") as scope:
            logits = self.model(x)
            return logits
