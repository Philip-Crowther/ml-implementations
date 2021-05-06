# perceptron.py - implementation of a perceptron
import matplotlib.pyplot as plt
import seaborn as sns


class Perceptron:
    def __init__(self, r=.5):
        self.bias = None
        self.x_weight = 0
        self.y_weight = 0
        self.r = r  # learning rate
        self.category = None
        self.x_train = None
        self.y_train = None

    def train(self, x_train, y_train, category):
        """training algorithm"""
        assert len(x_train) == len(y_train) == len(category), 'unequal numbers of values'
        self.x_train = x_train
        self.y_train = y_train
        self.category = category
        # TODO: training algorithm
        pass

    def predict(self, x):
        """uses trained perceptron to predict y values"""
        # TODO: predict values
        pass

    def accuracy(self, x_test, y_test):
        """returns an accuracy rating of a trained perceptron based off a test set"""
        # TODO: accuracy algorithm
        pass

    def visualize(self):
        """visualizes the training set and the perceptron"""
        assert self.x_train and self.y_train, 'error: no training data'
        # TODO: scatter-plot
        # TODO: line-plot
        pass


def main():
    pass


if __name__ == '__main__':
    main()
