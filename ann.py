#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_der(x):
    sig = sigmoid(x)
    return sig * (1-sig)

class HyperParameters:
    def __init__(self, alpha, momentum=0):
        self.alpha = alpha
        self.momentum = momentum

class ANN:
    """
        Artificial Neural Network

        Terms:
            y   :   linear output -> y = wx + b
            z   :   activated output -> activate(y)
    """

    def __init__(self, topology, hyperparams, activation, activation_der, epoch = 200):
        self.hyperparams = hyperparams
        self.activation = activation
        self.activation_der = activation_der
        self.epoch = epoch
        self._init_synapses(topology)

    def _init_synapses(self, topology):
        self.topology = topology
        # one bias per neuron (except input)
        self.biases = [ 2*np.random.random( (1, size) ).astype('f') - 1 for size in topology[1:] ]
        self.synapses = [ 2*np.random.random( size ).astype('f') - 1 for size in zip(topology[:-1], topology[1:]) ]

    def train_in_batch(self, X_train_whole, Y_train_whole, batch_size):
        """
            The entry point for training the ann.
            It slices the whole training set into batches.
        """
        i, k = 0, 0
        size = len(X_train_whole)
        costs = []
        for i, k in enumerate(range(0, size, batch_size)):
            cost =  self.train(X_train_whole[k : k + batch_size],
                               Y_train_whole[k : k + batch_size])
            print("Batch ==> {} ::: Cost ==> {} ".format(i, cost))
            costs.append(cost)
        return costs

    def train(self, X_train, Y_train):
        """
            mini-batch training
        """
        n = len(X_train)
        costs = []

        # to keep track of dw and db for using momentum
        delta_synapses = [ np.zeros(synapse.shape).astype('f') for synapse in self.synapses ]
        delta_biases = [ np.zeros(bias.shape).astype('f') for bias in self.biases ]
        for i in range(self.epoch):
            Z, cache_y, cache_z = self.feed_forward(X_train)
            cache_z = [X_train] + cache_z
            grad_synapses, grad_biases, cost = self.backpropagate(Y_train, cache_y, cache_z)

            # dw = - eta * gradient_w - momentum * previous_dw
            # w += dw
            for i, synapse in enumerate(self.synapses):
                delta_synapses[i] = -1/n * self.hyperparams.alpha * grad_synapses[i] \
                    - self.hyperparams.momentum * delta_synapses[i]
                delta_biases[i] = -1/n * self.hyperparams.alpha * np.sum(grad_biases[i], axis=0) \
                     - self.hyperparams.momentum * delta_biases[i]
                self.synapses[i] = synapse + delta_synapses[i]
                self.biases[i] = self.biases[i] + delta_biases[i]
            costs.append(cost)
        return np.mean(costs)

    def predict(self, X):
        Z, cache_y, cache_z = self.feed_forward(X)
        return Z

    def feed_forward(self, X):
        """
            Return final output along with all the activated output in each layer
        """
        inp = X
        cache_y = []
        cache_z = []
        for synapse, bias in zip(self.synapses, self.biases):
            Y = np.dot(inp, synapse) + bias
            cache_y.append(Y)
            Z = self.activation(Y)
            cache_z.append(Z)
            inp = Z
        return inp, cache_y, cache_z

    def backpropagate(self, Y_train, cache_y, cache_z):
        """
            Mighty backpropgation which is just reverse-mode differentiation.
            We apply chain rule from outer layer to inner layers.

            At outer layer:
                dJ/dWl =
                        ( delta = error * sigmoid_prime( Yl) )
                        * (previous layer activation (Zl-1) )

            At inner layer:
                dJ/dWl-1 = ( delta = delta * sigmoid_prime(Yl-1) * Wl)
                            * (previous layer activation (Zl-1))

            Return gradients and cost
        """
        grad_synapses = [ np.zeros(synapse.shape).astype('f') for synapse in self.synapses ]
        grad_biases = [ np.zeros(bias.shape).astype('f') for bias in self.biases ]
        Z = cache_z[-1]
        errors = self.cost_der(Y_train, Z)
        #cost = np.sum(errors**2) / len(errors)
        cost = self.calculate_cost(Y_train, Z)
        delta = errors * self.activation_der(cache_y[-1])

        grad_biases[-1] = delta
        grad_synapses[-1] = np.dot(cache_z[-2].T, delta)

        for l in range(2, len(self.topology)):
            Y = cache_y[-l]
            der = self.activation_der(Y)
            delta = np.dot(delta, self.synapses[-l+1].T) * der
            grad_biases[-l] = delta
            grad_synapses[-l] = np.dot(cache_z[-l-1].T, delta)
        return grad_synapses, grad_biases, cost

    def cost_der(self, target, predicted):
        #return self.lse_der(target, predicted)
        return self.cross_entropy_der(target, predicted)

    def calculate_cost(self, target, predicted):
        #return self.lse(target, predicted)
        return self.ace(target, predicted)

    def ace(self, target, predicted):
        """
            Average Cross Entropy
        """
        return -1/len(target) * np.sum(  target * np.log(predicted) + (1-target) * np.log( 1 - predicted) )

    def lse(self, target, predicted):
        """
            Least Squared Error
        """
        return 1/len(target) * np.sum( (target-predicted) ** 2 )


    def cross_entropy_der(self, target, predicted):
        return -( (target/predicted) - (1-target)/(1-predicted)  )

    def lse_der(self, target, predicted):
        """
            Return the partial derivative of cost function
                w.r.t final layer activation

            Cost function(J):  (1/2) * (target-predicted)**2
            So, dJ/dZ :
                (Z-T)
        """
        return -target + predicted


def test_ann():
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y_train = np.array([[0, 1, 1, 0]]).T
    topology = [2, 3, 1]
    hyperparams = HyperParameters(0.1)
    ann = ANN(topology, hyperparams, sigmoid, sigmoid_der, epoch=10000)
    #print(ann.biases)
    costs = ann.train(X_train, Y_train)
    print("Last cost => {}".format(costs[-1]))
    plt.plot(costs)
    plt.show()
    Z = ann.predict(np.array([0, 1]))
    print(Z)


def main():
    test_ann()

if __name__ == "__main__":
    main()

