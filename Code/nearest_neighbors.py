
import numpy as np
import tensorflow as tf

# https://gist.github.com/mbsariyildiz/34cdc26afb630e8cae079048eef91865
def pairwise_dist(A, B):
  """
  Computes pairwise distances between each elements of A and each elements of B.
  Args:
    A,    [m,d] matrix
    B,    [n,d] matrix
  Returns:
    D,    [m,n] matrix of pairwise distances
  """

  with tf.name_scope("PairwiseDist") as scope:
    # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)

    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    D = tf.sqrt(tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0))
  return D
'''
# GOAL:  Explore the other norms for K-NN
# ISSUE:  For large matrices, the L2 norm computed contains negative values
def pairwise_dist(X_test, X_train, ord = 2):
    #X_test = tf.Print(X_test, [tf.reduce_max(tf.abs(X_test)), tf.shape(X_test)], "X_test ")
    delta = tf.add(tf.expand_dims(X_test, 1), tf.expand_dims(tf.negative(X_train), 0))
    #delta = tf.Print(delta, [tf.reduce_max(tf.abs(delta)), tf.shape(delta)], "delta ")
    if ord == 2:
        square = tf.square(delta)
        #square = tf.Print(square, [tf.reduce_max(tf.abs(square)), tf.shape(square)], "square ")
        return tf.reduce_sum(square, reduction_indices = 2)
    elif ord == 1:
        return tf.reduce_sum(tf.abs(delta), reduction_indices = 2)
    elif ord == np.inf:
        return tf.reduce_max(tf.abs(delta), reduction_indices = 2)
'''

#Y is 1 for AE and -1 for Real Image
def knn(X_test, X, Y, K, bias = 0.0):
    dist = pairwise_dist(X_test, X)
    #dist = tf.Print(dist, [tf.reduce_min(dist), tf.shape(dist)], "dist ")
    _, indices = tf.nn.top_k(-1.0 * dist, k = K)
    votes = tf.reduce_mean(tf.gather(Y, indices), axis = 1) - bias #vote > 0 iff K-NN labels the point adversarial
    return votes
