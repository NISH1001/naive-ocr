#!/usr/bin/env python3

"""
    Some awesome garbage goes here for the sake of 'modularity'
"""

import config
import json
import matplotlib.pyplot as plt

def dump_shit(filename, topology, hyperparams, train_size, test_size,
              batch_size, epoch, result_train, result_test):
    print("Dumping shit... Cover your nose... :P ")
    data = []
    try:
        with open(filename, 'r') as f:
            data = json.loads(f.read())
    except FileNotFoundError:
        with open(filename, 'w') as f:
            json.dump(f, [], indent=4)

    to_dump = {}

    train = {
        'size': train_size,
        'accuracy': result_train.accuracy,
        'precision': result_train.precision,
        'recall': result_train.recall,
        'f1': result_train.f1
    } if result_train else {}

    test = {
        'size': test_size,
        'accuracy': result_test.accuracy,
        'precision': result_test.precision,
        'recall': result_test.recall,
        'f1': result_test.f1
    } if result_test else {}

    to_dump['topology'] = topology if topology else []
    to_dump['hyperparams'] = [hyperparams.learning_rate, hyperparams.momentum, hyperparams.reg_param] \
        if hyperparams \
        else []
    to_dump['batch_size'] = batch_size if batch_size else -1
    to_dump['epoch'] = epoch if epoch else -1
    to_dump['train'] = train
    to_dump['test'] = test

    data.append(to_dump)

    with open(filename, 'w') as f:
         json.dump(data, f, indent=4)

def plot_metrics(filename):
    accuracies_test = []
    accuracies_train = []
    with open(filename, 'r') as f:
        results = json.loads(f.read())
        print(len(results))
        for data in results:
            accuracies_train.append(data['train']['accuracy'])
            accuracies_test.append(data['test']['accuracy'])

    plt.plot(accuracies_train, color='b', label="Training Accuracy")
    plt.plot(accuracies_test, color='r', label="Test/Validation Accuracy")
    plt.legend()
    plt.show()


def main():
    #dump_shit(config.FILE_RESULT, 1,1,1,1,1,1)
    #dump_shit("data/result.json", None, None, None, None, None, None, None, None)
    plot_metrics(config.FILE_RESULT)


if __name__ == "__main__":
    main()

