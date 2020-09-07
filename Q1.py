import numpy as np
import matplotlib.pyplot as plt
import random


def activation(input, w, threshold, bias):
    return bias + np.matmul(input, w)
    # if np.matmul(input, w) >= threshold:
    #     return 1
    # else:
    #     return 0


def calculate_error(desired, output):
    return desired - output


def update_features(input, w, bias, error, learning_rate):
    w[0] = w[0] + learning_rate * input[0] * error
    w[1] = w[1] + learning_rate * input[1] * error
    bias = bias + learning_rate * error
    return w, bias


if __name__ == '__main__':
    threshold = 1.5
    learning_rate = 0.1

    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    inputs = np.array(inputs)
    desired = [0, 0, 1, 0]

    w = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
    print(w)
    e = [1]
    i = 0
    bias = random.uniform(-1, 1)
    error = 1
    while np.average(e) > 0.0005 or len(e) < 4000:
        for i in range(0, 4):
            output = activation(inputs[i], w, threshold, bias)
            error = calculate_error(desired[i], output)
            e.append(error)
            w, bias = update_features(inputs[i], w, bias, error, learning_rate)
    x, y = inputs.T
    plt.stem(e, use_line_collection=True)
    plt.show()
    plt.ylim(-0.1, 1.1)
    plt.scatter(x, y)
    plt.plot(np.linspace(0, 1, 2), w, 'k--')
    plt.show()
    print(w)
    print(len(e) / 4)
