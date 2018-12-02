from data_ingestor import DataIngestor
from metrics_tracker import MetricsTracker
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
from logistic_regression import LogisticRegression
import numpy as np


def main():
    test_linear_regression()
    test_logistic_regression()

def test_logistic_regression():
    data_ingestor = DataIngestor()
    metrics_tracker = MetricsTracker()
    train_x, train_y = data_ingestor.read_csv("train_india_diabetes.csv")
    print("Shape train_x: " + str(train_x.shape))
    print("Shape train_y: " + str(train_y.shape))
    

    logisticRegression = LogisticRegression()
    accuracies, costs = logisticRegression.train(train_x, train_y, nb_epochs=100000, batch_size=128, lr=0.001)
    test_y_logistic = logisticRegression.predict(train_x) > 0.5

    print(test_y_logistic)
    print("Accuracy: " + str(metrics_tracker.accuracy(train_y, test_y_logistic)))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col')
    ax1.plot(costs)
    ax2.plot(accuracies)
    plt.show()


def test_linear_regression():
    metrics_tracker = MetricsTracker()
    n_samples = 200
    x = np.linspace(0, 2 * np.pi, n_samples).reshape(n_samples, 1)
    noise = np.random.randn(x.shape[0]).reshape(n_samples, 1)
    y = -4*x**2 + 7  + noise*2.0
    y.reshape(n_samples, 1)
    linearRegression = LinearRegression()
    linearRegression.train(x, y, nb_epochs=5000, batch_size=128, lr=0.001)
    test_y_linear = linearRegression.predict(x)

    plt.scatter(x, y)
    plt.plot(x, test_y_linear)
    plt.show()

main()