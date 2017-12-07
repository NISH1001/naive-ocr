#!/usr/bin/env python3

import config
import numpy as np

from ann import HyperParameters
from ocr  import build_model, evaluate_model, convert_prob_to_label
from featurizer import featurize, one_hot_encoder
import matplotlib.pyplot as plt

def load_train(filename):
    print("Loading training data...")
    X_train = []
    Y_train = []
    with open(filename, 'r') as f:
        for line in f:
            splitted = line.split(',')
            try:
                label = int(splitted[0])
                vals = splitted[1:]
                Y_train.append(label)
                X_train.append(list(map(int, vals)))
            except ValueError:
                continue
    return X_train, Y_train

def load_test(filename):
    print("Loading test data...")
    X_test = []
    with open(filename, 'r') as f:
        for line in f.readlines()[1:]:
            vals = line.split(',')
            try:
                X_test.append(list(map(int, vals)))
            except ValueError:
                continue
    return X_test

def run():
    images, labels = load_train("data/train.csv")
    N = len(images)
    train_size = 41000
    eval_size = N-train_size

    print("shuffling...")
    dataset = list(zip(images, labels))
    np.random.shuffle(dataset)
    images, labels = zip(*dataset)

    X_train = np.array(images[0 : train_size])
    X_train = X_train/255
    Y_train = np.array(labels[0 : train_size])
    Y_train = one_hot_encoder(Y_train)
    X_eval = np.array(images[train_size:N])
    X_eval = X_eval/255
    Y_eval = np.array(labels[train_size:N])
    Y_eval = one_hot_encoder(Y_eval)


    hyperparams = HyperParameters(0.06, 0.5)
    topology = [X_train[0].shape[0], 100, 50, 10]
    batch_size = 100
    epoch = 25
    model, costs = build_model(topology, config.SIGMOID_SOFTMAX_CROSSENTROPY,
                               hyperparams, batch_size,
                               epoch, X_train, Y_train)
    plt.plot(costs)
    plt.show()

    result = evaluate_model(model, 10, X_eval, Y_eval)
    print(result)

    X_test = load_test("data/test.csv")
    X_test = np.array(X_test)
    X_test = X_test/255
    predicted = model.predict(X_test)
    predicted_label = convert_prob_to_label(predicted)
    predicted = np.ravel(predicted_label)
    print(predicted)
    np.savetxt("data/result", np.dstack((np.arange(1, predicted.size+1),predicted))[0],"%d,%d",header="ImageId,Label")

def main():
    run()

if __name__ == "__main__":
    main()

