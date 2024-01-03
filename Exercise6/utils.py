import numpy as np


def one_hot_encode(y: np.ndarray) -> np.ndarray:
    one_hot = np.zeros((len(y), len(np.unique(y))))
    for i, label in enumerate(y):
        one_hot[i, label] = 1
    return one_hot


def cross_entropy_loss(expected_dist: np.ndarray,
                       predicted_dist: np.ndarray) -> np.ndarray:
    array = np.array([-float(y) * np.log(float(x)) for x, y in
                      zip(predicted_dist, expected_dist)])
    return array


def cross_entropy_loss_derivative(expected_dist: np.ndarray,
                                  predicted_dist: np.ndarray) -> np.ndarray:
    array = np.array([x - y for x, y in zip(predicted_dist, expected_dist)])
    return array


def squared_error(expected_dist: np.ndarray,
                  predicted_dist: np.ndarray) -> np.ndarray:
    return np.array([pow(expected - predicted, 2) for expected, predicted in
                     zip(expected_dist, predicted_dist)])


def squared_error_derivative(expected_dist: np.ndarray,
                             predicted_dist: np.ndarray) -> np.ndarray:
    return 2 * (predicted_dist - expected_dist)


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    if filledLength != 100:
        bar = fill * filledLength + ">" + '-' * (length - filledLength - 1)
    else:
        bar = fill * filledLength + '-' * (length - filledLength)

    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
