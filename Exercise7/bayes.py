from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import math
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


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
                try:
                    value *= self.likelihoods[label][i][att_value]
                except KeyError:
                    value *= 0
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

                attribute_mean = np.mean(attribute)
                attribute_std = np.std(attribute)
                attribute_dict = {"mean": attribute_mean, "std": attribute_std}

                label_dict[i] = attribute_dict
            self.likelihoods[label] = label_dict

    @staticmethod
    def normal_dist(x, mean, std):
        if std == 0:
            return 0
        else:
            prob_density = (np.pi * std) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        return prob_density

    def predict(self, sample):
        predictions = {}
        for label, prior in self.priors.items():
            value = 1
            for i, att_value in enumerate(sample):
                att_mean = self.likelihoods[label][i]["mean"]
                att_std = self.likelihoods[label][i]["std"]
                value *= self.normal_dist(att_value, att_mean, att_std)
            predictions[label] = value

        return max(predictions, key=predictions.get)


def generate_prediction_tuples(classifier, x_data: np.ndarray, y_data: np.ndarray):
    prediction_and_label = []
    for data, label in zip(x_data, y_data):
        pred = classifier.predict(data)
        prediction_and_label.append((pred, label))
    return prediction_and_label


def measure_accuracy(classifier, x_data: np.ndarray, y_data: np.ndarray) -> float:
    predictions_list = generate_prediction_tuples(classifier, x_data, y_data)
    predicted_successfully = sum([1 for x, y in predictions_list if x == y])
    return predicted_successfully / len(predictions_list)


tries = 100
iris = load_iris()
x = iris.data
y = iris.target

accuracies = []
split_values = []
for test_size_mult in range(1, 20):
    naive_temp_accuracy = []
    gauss_temp_accuracy = []
    for i in range(tries):
        x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x, y, test_size=0.5*test_size_mult/10)
        naive_1 = NaiveBayes()
        gauss_1 = GaussianNaiveBayes()

        naive_1.build_classifier(x_train_1, y_train_1)
        gauss_1.build_classifier(x_train_1, y_train_1)

        discrete_x_test_1 = naive_1.data_discretization(x_test_1)
        naive_temp_accuracy.append(measure_accuracy(naive_1, discrete_x_test_1, y_test_1))
        gauss_temp_accuracy.append(measure_accuracy(gauss_1, x_test_1, y_test_1))

    accuracies.append((sum(naive_temp_accuracy)/tries, sum(gauss_temp_accuracy)/tries))
    split_values.append(0.5*test_size_mult/10)

plt.plot(split_values,[x[0] for x in accuracies],'o-', label=f"Naive Bayes")
plt.plot(split_values,[x[1] for x in accuracies],'o-', label=f"Gaussian Naive Bayes")


plt.grid()
plt.legend()
plt.title("Precyzja przewidywań zbioru Iris dla różnej wielkości zbioru treningowego")
plt.ylabel("Skuteczność [%]")
plt.xlabel("Wielkość zbioru testowego w porównaniu do całości zbioru")
plt.show()