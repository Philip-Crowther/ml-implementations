# perceptron.py - implementation of a perceptron
import matplotlib.pyplot as plt
import seaborn as sns


class Perceptron:
    def __init__(self, r=.25):
        self.w = [0, 0, 0]
        self.r = r  # learning rate
        self.data = None

    def train(self, data):
        """training algorithm"""
        self.data = data
        # TODO: training algorithm
        for datum in data:
            for i in range(len(datum)-1):
                # update the weights for each dimension of the data
                self.w[i] = self.w[i] + self.r * datum[-1] * x # datum[-1] is the classification for a piece of data
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
        assert self.x_train, 'error: no training data'
        sns.scatterplot(x=self.x_train, y=self.y_train, hue=self.category)
        # TODO: line-plot
        pass


def main():
    pass


if __name__ == '__main__':
    main()
