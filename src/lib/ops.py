import numpy as np
import scipy as sp
import tensorflow as tf

def conv_via_embedding(X, W, dilation=1):
    if type(X) is list:
        Xprev, Xcur, Xnext = X 
    else:
        Xprev = X[:-2*dilation]
        Xcur  = X[dilation:-dilation]
        Xnext = X[2*dilation:]
     
    conv_out = tf.nn.embedding_lookup(W[0, :, :], Xprev) + tf.nn.embedding_lookup(W[1, :, :], Xcur) + tf.nn.embedding_lookup(W[2, :, :], Xnext)

    return conv_out

def conv_via_matmul(X, W, dilation=1):
    if type(X) is list:
        Xprev, Xcur, Xnext = X
    else:   
        Xprev = X[:-2*dilation, :]
        Xcur  = X[dilation:-dilation, :]
        Xnext = X[2*dilation:, :] 

    Xconcat = tf.concat([Xprev, Xcur, Xnext], axis=1)
 
    conv_out = tf.matmul(Xconcat, W)    

    return conv_out


def int_shape(x):
    return list(map(int, x.get_shape()))

def concat_relu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = 1
    return tf.nn.relu(tf.concat([x, -x], axis))


