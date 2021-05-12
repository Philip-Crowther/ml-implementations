# perceptron.py - implementation of a perceptron
import random as r
from math import e


def binary_step(z):
    """performs binary step activation function"""
    a = 1 if z > 0 else -1
    return a


def sigmoid_activation(z):
    """performs sigmoid activation function"""
    a = 1 / (1 + e ** -z)
    return a


class Perceptron:
    def __init__(self):
        self.w = None

    def train(self, data, r=.5, cycles=150):
        """training algorithm"""
        # initialize weights, first weight (self.w[0]) is the model's bias
        self.w = [0 for _ in range(len(data[0]))]
        # store best encountered list of weights and its accuracy to avoid losing a more accurate set of weights that we encounter
        best_w, best_accuracy = self.w, 0
        # train the perceptron until it either reaches a certain accuracy or completes a certain number of trainings
        while cycles:
            # find current accuracy
            accuracy = self.test_accuracy(data)
            # update best accuracy and best weights if the current weight and accuracy are better
            if accuracy > best_accuracy:
                best_accuracy, best_w = accuracy, self.w
            # countdown number of cycles left
            cycles -= 1
            # test each item of data for its accuracy
            for datum in data:
                for i in range(1, len(datum)):
                    # update the weights for each dimension of the data if this piece of data was mis-predicted
                    if datum[0] == 1 and self.predict(datum) <= 0:
                        self.w[i] = self.w[i] + r * datum[0] * datum[i]  # datum[0] is the classification for a piece of data
                    elif datum[0] == -1 and self.predict(datum) > 0:
                        self.w[i] = self.w[i] + r * datum[0] * datum[i]
        # set this instances weight as the best encountered
        self.w = best_w if best_accuracy > self.test_accuracy(data) else self.w

    def predict(self, datum):
        """predicts for a single data point"""
        return self.w[0] + sum([self.w[i] * datum[i] for i in range(1, len(datum))])

    def classify(self, data, activation_func=binary_step):
        """classifies a set of data and sends it through an activation function"""
        # storage for each piece of classified data
        classified_data = []
        # cycle through each item of data
        for i in range(len(data)):
            # find sum of all the weighted items in this piece of data
            z = self.w[0] + sum([self.w[j + 1] * data[i][j] for j in range(len(data[i]))])
            # apply activation function
            classification = activation_func(z)
            # insert classified data into storage
            classified_data.append([classification] + data[i])
        return classified_data

    def test_accuracy(self, test_data):
        """returns an accuracy rating of a trained perceptron based off a test set"""
        correct = 0
        for datum in test_data:
            if datum[0] == 1 and self.predict(datum) > 0:
                correct += 1
            elif datum[0] == -1 and self.predict(datum) <= 0:
                correct += 1
        return correct / len(test_data)

    # TODO: figure out how to store activation functions as class methods


def main():
    """testing"""

    # 2D
    bias = 5
    data = [[r.randint(-50, 50), r.randint(-50, 50)] for _ in range(400)]
    data = [[1] + datum if datum[0] + bias > datum[1] else [-1] + datum for datum in data]

    accuracy_test_data = [[r.randint(-50, 50), r.randint(-50, 50)] for _ in range(100)]
    accuracy_test_data = [[1] + datum if datum[0] + bias > datum[1] else [-1] + datum for datum in accuracy_test_data]

    p = Perceptron()
    p.train(data)
    print('accuracy with original dataset: ', p.test_accuracy(data))
    print('accuracy for accuracy_test_data', p.test_accuracy(accuracy_test_data))


if __name__ == '__main__':
    main()
