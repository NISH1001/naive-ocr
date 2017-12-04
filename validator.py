#!/usr/bin/env python3

import numpy as np

class Result:
    def __init__(self, confusion_matrix, accuracy, precision, recall, f1):
        self.confusion_matrix =  confusion_matrix
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1

class Validator:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes

    def calculate_accuracy(self, target_label, predicted_label):
        matched = np.sum(target_label == predicted_label)
        return matched / len(target_label)

    def calculate_metrics(self, target_label, predicted_label):
        #n = len(target_label)
        result = {}
        cm = np.zeros((self.num_classes, self.num_classes))
        precisions = np.zeros(self.num_classes)
        recalls = np.zeros(self.num_classes)
        for t, p in zip(target_label, predicted_label):
            cm[t][p] += 1

        tp = np.diag(cm)
        fn = np.sum(cm, axis=1) - tp
        fp = np.sum(cm, axis=0) - tp

        for i in range(self.num_classes):
            p_denom = tp[i] + fp[i]
            r_denom = tp[i] + fn[i]
            precisions[i] = 0 if p_denom == 0 else tp[i]/p_denom
            recalls[i] = 0 if r_denom == 0 else tp[i]/r_denom

        return cm, precisions, recalls

def main():
    t = np.array([1, 3, 2, 1, 3, 4, 0])
    p = np.array([1, 2, 2, 1, 3, 3, 0])
    validator = Validator(5)
    cm, precisions, recalls = validator.calculate_metrics(t, p)
    print(cm, np.mean(precisions), np.mean(recalls))

if __name__ == "__main__":
    main()

