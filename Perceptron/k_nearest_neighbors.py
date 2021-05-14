# k_nearest_neighbors.py - implementation of the k-nearest neighbors algorithm
from math import sqrt
from bisect import bisect
from collections import Counter


class KNeighbors:
    def __init__(self, data=None):
        """initializes with the option to add training data upon initialization"""
        if data is None:
            data = []
        # last item in a piece of data is its classification
        self.data = data

    def train(self, training_data):
        """take in training data"""
        self.data += training_data

    def classify(self, testing_data, k):
        """perform the k-nearest neighbors classification algorithm on a set of data"""
        # to store any classified data points
        classified_data = []
        # find k-nearest neighbors for each unclassified data point in data
        for point1 in testing_data:
            # sorted container for distances between this point and every training point
            neighbors = []
            for point2 in self.data:
                # find distance between the two points
                distance = self.find_distance(point1, point2)
                # create tuple for storage formatted: (distance, classification)
                neighbor = (distance, point2[-1])
                # find where neighbor would be located in order to maintain neighbors' sorted status
                index = bisect(neighbors, neighbor)
                # update neighbors
                neighbors.insert(index, neighbor)
            # create list of the classifications of the k-nearest neighbors
            k_neighbors = neighbors[:k][1]
            # create a frequency counter for the classifications
            classification_counter = Counter(k_neighbors)
            # find most common classification
            classification = classification_counter.most_common()
            classified_data.append([point1 + [classification]])
        # return list of points in same order with their classifications as the last item in each data point's array
        return classified_data

    @staticmethod
    def find_distance(p1, p2):
        """finds euclidean distance between two points p1 and p2"""
        return sqrt(sum([(p1[i] + p2[i]) ** 2 for i in range(len(p1))]))


def main():
    """main function to hold testing"""
    pass


if __name__ == '__main__':
    main()
