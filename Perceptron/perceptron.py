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
    """
    Perceptron Class
    ================

    Attributes
    ----------
    w:  stores the Percepton's bias in w[0] and weights for each dimension of the data in the following spots:
        each index after 0 in w, w[i], is the corresponding weight for the value at the same index in each data point,
        data_point[i]. w is initialized as None, its length is determined when the training values are input to allow
        the algorithm to train on datasets of any dimensionality

    Methods
    -------
    train(data, r=.5, cycle=150):   perceptron training algorithm, sets the weights for this instance of the Perceptron class
    predict(datum):                 predicts classification for a single data point based off of weights in self.w
    classify(data, activation_func=binary_step): classifies a set of data using specified activation function
    test_accuracy(test_data):       will test Perceptron's current accuracy on a data set with known classifications

    """
    def __init__(self):
        """constructs holder for Perceptron's weights"""
        self.w = None

    def train(self, data, r=.5, cycles=150):
        """
        perceptron training algorithm, sets the weights for this instance of the Perceptron class

        Args:
            data:   array of n+1 sized arrays storing n-dimensional data points with their binary classification stored
                    as data[0] [classification, x1, x2, x3, ...]
            r:      learning rate (defaults at .5)
            cycles: number of desired training cycles (defaults at 150)
        """
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
        """predicts classification for a single data point based off of weights in self.w """
        return self.w[0] + sum([self.w[i] * datum[i] for i in range(1, len(datum))])

    def classify(self, data, activation_func=binary_step):
        """classifies a set of data using specified activation function"""
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
        """
        tests Perceptron's current accuracy on a data set with known classifications

        Args:
            test_data:  data set with known correct classification to be tested against the perceptron's predictions
                        each data point is an array formatted with the classification at data_point[0]
        Return:
            percentage of data points in the passed array the the perceptron classifies correctly in decimal format
        """
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

    # 3D

    p = Perceptron()
    p.train(data)
    print('accuracy with original dataset: ', p.test_accuracy(data))
    print('accuracy for accuracy_test_data', p.test_accuracy(accuracy_test_data))


if __name__ == '__main__':
    main()
