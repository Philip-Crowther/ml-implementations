# perceptron.py - implementation of a perceptron
import random as r


class Perceptron:
    def __init__(self, r=.25):
        self.w = None
        self.r = r  # learning rate

    def train(self, data, goal=1.0, cycles=100):
        """training algorithm"""
        self.w = [0 for _ in range(len(data))]
        # TODO: tune/troubleshoot training algorithm
        # train the perceptron until it either reaches a certain accuracy or completes a certain number of trainings
        while cycles and self.accuracy(data) < goal:  # while too many mis-classifications
            cycles -= 1
            for datum in data:
                for i in range(1, len(datum)):
                    # update the weights for each dimension of the data if it is predicted incorrectly
                    if self.w[0] + sum([self.w[i] * datum[i] for i in range(1, len(datum))]) > 0:
                        if datum[0] == 1:
                            self.w[i] = self.w[i] + self.r * datum[0] * datum[
                                i]  # datum[0] is the classification for a piece of data
                    elif datum[0] == -1:
                        self.w[i] = self.w[i] + self.r * datum[0] * datum[i]
        pass

    def predict(self, data):
        """categorizes a set of data"""
        return [[1] + data[i] if self.w[0] + sum([self.w[j + 1] * data[i][j] for j in range(len(data[i]))]) > 0 else [-1] + data[i] for i in range(len(data))]

    def accuracy(self, test_data):
        """returns an accuracy rating of a trained perceptron based off a test set"""
        # TODO: tune/troubleshoot accuracy algorithm
        correct = 0
        for datum in test_data:
            if self.w[0] + sum([datum[i] * self.w[i] for i in range(1, len(datum))]) > 0:
                if datum[0] == 1:
                    correct += 1
            elif datum[0] == -1:
                correct += 1
        return correct / len(test_data)


def main():
    # going to generate a data set here to test the perceptron with

    # 2D
    data = [[r.randint(-50, 50), r.randint(-50, 50)] for _ in range(100)]
    data = [[[1] + datum if datum[0] > datum[1] else [0] + datum for datum in data]]
    accuracy_test_data = [[r.randint(-50, 50), r.randint(-50, 50)] for _ in range(100)]
    accuracy_test_data = [[[1] + datum if datum[0] > datum[1] else [0] + datum for datum in accuracy_test_data]]
    predict_data =[[r.randint(-50, 50), r.randint(-50, 50)] for _ in range(100)]

    p = Perceptron()
    p.train(data)
    print('accuracy with original dataset: ', p.accuracy(data))
    print('accuracy for accuracy_test_data', p.accuracy(accuracy_test_data))
    predict_data = p.predict(predict_data)
    print('accuracy with test set: ', p.accuracy(predict_data))


if __name__ == '__main__':
    main()
