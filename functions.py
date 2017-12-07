"""
    Module to accumulate every activation functions and cost function
"""

import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_der(x):
    sig = sigmoid(x)
    return sig * (1-sig)

def tanh(x):
    return np.tanh(x)

def tanh_der(x):
    y = tanh(x)
    return 1 -  y*y

def softmax(x):
    e = np.exp(x - np.amax(x))
    return e / np.sum(e)

def softmax_der(x):
    y = softmax_generalized(x)
    return y * (1-y)

def softmax_generalized(x, theta = 1.0, axis = 1):
    """
        With axis=1, apply softmax for each row
    """
    # make X at least 2d
    y = np.atleast_2d(x)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    p = y / ax_sum

    # flatten if X was 1D
    if len(x.shape) == 1: p = p.flatten()

    return p

def ace(target, predicted):
    """
        Average Cross Entropy
    """
    return -1/len(target) * np.sum(  target * np.log(predicted) + (1-target) * np.log( 1 - predicted) )

def cross_entropy_der(target, predicted):
    #return -( (target/predicted) - (1-target)/(1-predicted)  )
    return - (target - predicted) / ( predicted * (1-predicted) )

def lse(target, predicted):
    """
        Least Squared Error
    """
    return 1/len(target) * np.sum( (target-predicted) ** 2 )

def lse_der(target, predicted):
    """
        Return the partial derivative of cost function
            w.r.t final layer activation

        Cost function(J):  (1/2) * (target-predicted)**2
        So, dJ/dZ :
            (Z-T)
    """
    return -target + predicted
