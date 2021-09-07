# naive_bayes_classifier.py - implementation of the naive bayes classification algorithm
from collections import Counter


class NaiveBayesClassifier:
    def __init__(self):
        """
        Naive Bayes Classifier Class
        =====================

        Attributes:
        -------------
        proposition_priors:     for each possible classification, the probability of any given data point being
                                the afforementiond classification is stored in this dictionary

                                {keys = classification: values = probability}

        evidence_priors:        for each feature, the probability of each possible value is stored here;
                                evidence_priors[x] is a dict of feature x's values

                                [   {x0_value0: prob, x0_value1: prob, x0_value2: prob,...},
                                    {x1_value0: prob, x1_value1:prob},
                                    {x2_value0: prob, x2_value1: prob},
                                    ...]

        likelihoods:            for each feature, we store its likelihood given a specific class here;
                                likelihoods[x] is a dict for x's likelihoods per class

                                [   {   x0_possible_value0: { classA: classA_probability, classB: classB_probability},
                                        x0_possible_value1: { classA: classA_probability, classB: classB_probability},
                                        x0_possible_value: { classA: classA_probability, classB: classB_probability}
                                        },

                                        x1_possible_value0: { classA: classA_probability, classB: classB_probability},
                                        x1_possible_value1: { classA: classA_probability, classB: classB_probability},
                                        },
                                        ...]


        # below are two different representations of bayes' theorem to clarify on the language used:
                            (A is called the proposition and B is called the evidence)

            P(A|B)      =  P(B|A)  *  P(A)  /  P(B)

            posterior   =  likelihood  *  proposition prior  /  evidence prior

        # for the scenario of a naive bayes' classifier: the evidence is a range of values (x1, x2, x3...)



        Methods:
        -------------
        fit(x_train, y_train)       training algorithm
        predict(x_test)             classification function
        """
        self.proposition_priors = {}
        self.evidence_priors = []
        self.likelihoods = []

    def fit(self, x_train, y_train):
        self.generate_proposition_priors(y_train)
        self.generate_evidence_priors(x_train)
        self.generate_likelihoods()

    def generate_proposition_priors(self, y_classifications):
        """
        takes a list of x number of classifications and returns a dictionary of the probability
        that any given data point in the list is any given classification
        """
        # get count of each classification
        counts = Counter(y_classifications)
        total_counts = len(y_classifications)
        # calculate and store probability of each classification
        for classification in counts:
            self.proposition_priors[classification] = counts[classification] / total_counts

    def generate_evidence_priors(self, x):
        """
        generates the evidence priors for each possible value of each feature of the model
        """
        number_of_data_points = len(x)
        number_of_features = len(x[0])
        # generate dictionaries for each feature to store its values
        self.evidence_priors = [{} for _ in range(number_of_features)]
        # go through each feature and get count of each value to calculate probability
        for feature in range(number_of_features):
            count_of_feature_values = Counter([x[i][feature] for i in range(number_of_data_points)])
            # store probability of each value in corresponding feature dictionary
            for possible_value in count_of_feature_values.keys():
                self.evidence_priors[feature][possible_value] = count_of_feature_values[possible_value] / number_of_data_points


    def generate_likelihoods(self):
        pass

    def predict(self, x_test):
        pass


def main():
    pass


if __name__ == "__main__":
    main()
