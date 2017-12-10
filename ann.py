#!/usr/bin/env python3

import config
import matplotlib.pyplot as plt
import numpy as np

class HyperParameters:
    def __init__(self, learning_rate, momentum=0, learning_rate_decay=0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.learning_rate_decay = learning_rate_decay

class ANN:
    """
        Artificial Neural Network

        Terms:
            y   :   linear output -> y = wx + b
            z   :   activated output -> activate(y)
    """

    def __init__(self, topology, config, hyperparams, epoch = 200):
        self.config = config
        self.hyperparams = hyperparams
        self.epoch = epoch
        self._init_synapses(topology)

    def _init_synapses(self, topology):
        self.topology = topology
        # one bias per neuron (except input)
        self.biases = [ 2*np.random.random( (1, size) ).astype('f') - 1 for size in topology[1:] ]
        self.synapses = [ 2*np.random.random( size ).astype('f') - 1 for size in zip(topology[:-1], topology[1:]) ]

        # to keep track of dw and db for using momentum
        self.delta_synapses = [ np.zeros(synapse.shape).astype('f') for synapse in self.synapses ]
        self.delta_biases = [ np.zeros(bias.shape).astype('f') for bias in self.biases ]

    def train_in_batch(self, X_train_whole, Y_train_whole, batch_size):
        """
            The entry point for training the ann.
            It slices the whole training set into batches.
        """
        size = len(X_train_whole)
        costs = []
        for i, k in enumerate(range(0, size, batch_size)):
            print("Learning rate ==> {}".format(self.hyperparams.learning_rate))
            cost =  self.train(X_train_whole[k : k + batch_size],
                               Y_train_whole[k : k + batch_size])
            print("Batch ==> {} ::: Cost ==> {} ".format(i, cost))

            # decay learning rate after each epoch (mini batch train)
            alpha = self.hyperparams.learning_rate
            self.hyperparams.learning_rate = alpha**2 / ( alpha + alpha* self.hyperparams.learning_rate_decay )
            #self.hyperparams.learning_rate = alpha / (1 + self.hyperparams.learning_rate_decay * i)
            costs.append(cost)
        return costs

    def train(self, X_train, Y_train):
        """
            mini-batch training
        """
        n = len(X_train)
        costs = []

        mu = self.hyperparams.momentum
        lr = self.hyperparams.learning_rate

        for i in range(self.epoch):
            Z, cache_y, cache_z = self.feed_forward(X_train)
            cache_z = [X_train] + cache_z
            grad_synapses, grad_biases, cost = self.backpropagate(Y_train, cache_y, cache_z)

            # dw = - lr * gradient_w + mu * previous_dw => standard momentum
            # w += dw
            # Also, v_prev = v
            # v = mu * v - lr * grad
            # w += -mu * v_prev + (1+mu) * v => Neterov Momentum => mostly used
            #
            for i, synapse in enumerate(self.synapses):
                delta_synapses_prev = self.delta_synapses[i][:]
                delta_biases_prev = self.delta_biases[i][:]
                self.delta_synapses[i] = -1/n * lr * grad_synapses[i] \
                    + mu * self.delta_synapses[i]
                self.delta_biases[i] = -1/n * lr * np.sum(grad_biases[i], axis=0) \
                     + mu * self.delta_biases[i]
                self.synapses[i] = synapse \
                            + (1 + mu) * self.delta_synapses[i] \
                            - mu * delta_synapses_prev
                self.biases[i] = self.biases[i] \
                            + (1 + mu) * self.delta_biases[i] \
                            - mu * delta_biases_prev
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

        # process for hidden layers
        for synapse, bias in zip(self.synapses[:-1], self.biases[:-1]):
            Y = np.dot(inp, synapse) + bias
            cache_y.append(Y)
            Z = self.config.activation_hidden(Y)
            cache_z.append(Z)
            inp = Z

        # process for output layer
        Y = np.dot(inp, self.synapses[-1]) + self.biases[-1]
        cache_y.append(Y)
        Z = self.config.activation_output(Y)
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
        delta = errors * self.config.activation_der_output(cache_y[-1])

        grad_biases[-1] = delta
        grad_synapses[-1] = np.dot(cache_z[-2].T, delta)

        for l in range(2, len(self.topology)):
            Y = cache_y[-l]
            der = self.config.activation_der_hidden(Y)
            delta = np.dot(delta, self.synapses[-l+1].T) * der
            grad_biases[-l] = delta
            grad_synapses[-l] = np.dot(cache_z[-l-1].T, delta)
        return grad_synapses, grad_biases, cost

    def cost_der(self, target, predicted):
        return self.config.cost_func_der(target, predicted)

    def calculate_cost(self, target, predicted):
        return self.config.cost_func(target,predicted)

def test_ann():
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y_train = np.array([[0, 1, 1, 0]]).T
    topology = [2, 3, 1]
    hyperparams = HyperParameters(0.1)
    ann = ANN(topology, hyperparams, config.SIGMOID_SOFTMAX_CROSSENTROPY, epoch=10000)
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

