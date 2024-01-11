from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import math
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)


class NaiveBayes:
    def __init__(self):
        self.priors = {}
        self.likelihoods = {}

    def build_classifier(self, train_features, train_classes):
        discrete_xtrain = self.data_discretization(train_features)
        labels = np.unique(train_classes)
        labels_count = Counter(train_classes)

        for label in labels:
            self.priors[label] = labels_count[label] / len(train_classes)

            label_dict = {}
            mask = train_classes == label
            for i in range(discrete_xtrain.shape[1]):
                attribute = discrete_xtrain[mask, i]
                attribute_counter = Counter(attribute)
                attribute_dict = {x: 0 for x in np.unique(discrete_xtrain)}
                for unique_value, number in attribute_counter.items():
                    print(number, labels_count[label])
                    attribute_dict[unique_value] = number / labels_count[label]

                label_dict[i] = attribute_dict
            self.likelihoods[label] = label_dict

    @staticmethod
    def data_discretization(data: np.ndarray) -> np.ndarray:
        intervals = 4

        discrete_array = np.zeros(data.shape)
        for i in range(data.shape[1]):
            min_value = np.min(data[:, i])
            max_value = np.max(data[:, i])
            interval = (max_value - min_value) / intervals
            for j in range(data.shape[0]):
                discrete_array[j, i] = min((data[j, i] - min_value) // interval,
                                           intervals - 1)

        return discrete_array

    def predict(self, sample):
        predictions = {}
        for label, prior in self.priors.items():
            value = 1
            for i, att_value in enumerate(sample):
                value *= self.likelihoods[label][i][att_value]
            predictions[label] = value

        return max(predictions, key=predictions.get)


class GaussianNaiveBayes:
    def __init__(self):
        self.priors = {}
        self.likelihoods = {}

    def build_classifier(self, train_features, train_classes):
        labels = np.unique(train_classes)
        labels_count = Counter(train_classes)

        for label in labels:
            self.priors[label] = labels_count[label] / len(train_classes)

            label_dict = {}
            mask = train_classes == label
            for i in range(train_features.shape[1]):
                attribute = train_features[mask, i]
                attribute_dict = {x: 0 for x in np.unique(train_features)}

                attribute_mean = np.mean(attribute)
                attribute_std = np.std(attribute)

                for value in np.unique(attribute):
                    attribute_dict[value] = self.normal_dist(value,
                                                             attribute_mean,
                                                             attribute_std)

                label_dict[i] = attribute_dict
            self.likelihoods[label] = label_dict

    @staticmethod
    def normal_dist(x, mean, std):
        prob_density = (np.pi * std) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        return prob_density

    def predict(self, sample):
        predictions = {}
        for label, prior in self.priors.items():
            value = 1
            for i, att_value in enumerate(sample):
                value *= self.likelihoods[label][i][att_value]
            predictions[label] = value

        return max(predictions, key=predictions.get)

bayes = GaussianNaiveBayes()
bayes.build_classifier(x_train, y_train)
#x_test = bayes.data_discretization(x_test)
positive = 0
for data, label in zip(x_test, y_test):
    pred = bayes.predict(data)
    print(pred, label)
    if pred == label:
        positive += 1

print(f"Accuracy: {(positive/len(y_test))*100:.2f}%")