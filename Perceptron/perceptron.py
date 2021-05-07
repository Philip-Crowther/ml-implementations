# perceptron.py - implementation of a perceptron
import matplotlib.pyplot as plt
import seaborn as sns


class Perceptron:
    def __init__(self, r=.25):
        self.w = None
        self.r = r  # learning rate
        self.data = None

    def train(self, data, goal=1.0, cycles=100):
        """training algorithm"""
        self.data = data
        self.w = [0 for _ in range(len(data))]
        # TODO: training algorithm
        # train the perceptron until it either reaches a certain accuracy or completes a certain number of trainings
        while cycles and self.accuracy() < goal:  # while too many mis-classifications
            cycles -= 1
            for datum in data:
                for i in range(1, len(datum)):
                    # update the weights for each dimension of the data if it is predicted incorrectly
                    if self.w[0] + sum([self.w[i] * datum[i] for i in range(1, len(datum))]) > 0:
                        if datum[0] == 1:
                            self.w[i] = self.w[i] + self.r * datum[0] * datum[i] # datum[0] is the classification for a piece of data
                    elif datum[0] == -1:
                        self.w[i] = self.w[i] + self.r * datum[0] * datum[i]
        pass

    def predict(self, x):
        """uses trained perceptron to predict y values"""
        # TODO: predict values
        pass

    def accuracy(self, test_data=self.data):
        """returns an accuracy rating of a trained perceptron based off a test set"""
        # TODO: accuracy algorithm
        correct = 0
        for datum in test_data:
            if self.w[0] + sum([datum[i] * self.w[i] for i in range(1, len(datum))]) > 0:
                if datum[0] == 1:
                    correct += 1
            elif datum[0] == -1:
                correct += 1
        return correct / len(test_data)


def main():
    pass


if __name__ == '__main__':
    main()
