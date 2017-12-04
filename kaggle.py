#!/usr/bin/env python3

import numpy as np

from ann import HyperParameters, ANN, sigmoid, sigmoid_der
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
    X_train = np.array(images[0 : train_size])
    Y_train = np.array(labels[0 : train_size])
    Y_train = one_hot_encoder(Y_train)
    X_eval = np.array(images[train_size:N])
    Y_eval = np.array(labels[train_size:N])
    Y_eval = one_hot_encoder(Y_eval)


    hyperparams = HyperParameters(0.03, 0.01)
    topology = [X_train[0].shape[0], 300, 10]
    batch_size = 200
    epoch = 50
    model, costs = build_model(topology, hyperparams, batch_size, epoch, X_train, Y_train)
    plt.plot(costs)
    plt.show()

    result = evaluate_model(model, 10, X_eval, Y_eval)

    print("Confusion Matrix ==> {}".format(result['confusion_matrix']))
    print("Accuracy ==> {}".format(result['accuracy']))
    print("Precision per class ==> {}".format(result['precision_per_class']))
    print("Average Precision ==> {}".format(result['precision']))
    print("Recall per class ==> {}".format(result['recall_per_class']))
    print("Average Recall ==> {}".format(result['recall']))
    print("F1 Score ==> {}".format(result['f1']))

    X_test = load_test("data/test.csv")
    X_test = np.array(X_test)
    predicted = model.predict(X_test)
    predicted_label = convert_prob_to_label(predicted)
    predicted = np.ravel(predicted_label)
    print(predicted)
    np.savetxt("data/result", np.dstack((np.arange(1, predicted.size+1),predicted))[0],"%d,%d",header="ImageId,Label")

def main():
    run()

if __name__ == "__main__":
    main()

