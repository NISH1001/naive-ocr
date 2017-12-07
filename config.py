"""
    Module for configuring our ANN
"""
from functions import *

class Config:
    def __init__(self, activation_hidden, activation_der_hidden,
                 activation_output, activation_der_output,
                 cost_func, cost_func_der):
        self.activation_hidden = activation_hidden
        self.activation_der_hidden = activation_der_hidden
        self.activation_output = activation_output
        self.activation_der_output = activation_der_output
        self.cost_func = cost_func
        self.cost_func_der = cost_func_der

SIGMOID_SOFTMAX_CROSSENTROPY = Config(
    sigmoid,
    sigmoid_der,
    softmax_generalized,
    softmax_der,
    ace,
    cross_entropy_der
)

SIGMOID_CROSSENTROPY = Config(
    sigmoid,
    sigmoid_der,
    sigmoid,
    sigmoid_der,
    ace,
    cross_entropy_der
)

SIGMOID_LSE = Config(
    sigmoid,
    sigmoid_der,
    sigmoid,
    sigmoid_der,
    lse,
    lse_der
)
