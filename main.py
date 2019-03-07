from data_ingestor import DataIngestor
from metrics_tracker import MetricsTracker
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
from logistic_regression import LogisticRegression
from linear_regression_mle import LinearRegressionMLE
from linear_regression_map import LinearRegressionMAP
from bayesian_linear_regression import BayesianLinearRegression
from PCA import PCA
import numpy as np


def main():
    #test_linear_regression()
    # test_logistic_regression()
   #test_linear_regression()
    #test_linear_regression_mle_simple()
    #test_linear_regression_mle_poly()
    #test_linear_regression_map_simple()
    #test_linear_regression_map_mle_poly()
    #test_bayesian_linear_regression()
    test_pca()

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
    y = -4*x + 7  + noise*2.0
    y.reshape(n_samples, 1)
    linearRegression = LinearRegression()
    metrics_tracker.profile(linearRegression.train, x, y, 1000, 128, 0.01)
    test_y_linear = metrics_tracker.profile(linearRegression.predict, x)
    err = metrics_tracker.mean_squared_error(test_y_linear, y)
    print("Squared Mean Error: " + str(err))

    plt.scatter(x, y)
    plt.plot(x, test_y_linear)
    plt.show()

def test_linear_regression_mle_simple():
    print("Test Linear Regression MLE:  ")
    metrics_tracker = MetricsTracker()
    n_samples = 200
    x = np.linspace(0, 2 * np.pi, n_samples).reshape(n_samples, 1)
    noise = np.random.randn(x.shape[0]).reshape(n_samples, 1)
    y = -4*x + 7  + noise*2.0
    y = y.reshape(n_samples, 1)

    linearRegressionMLE = LinearRegressionMLE()
    metrics_tracker.profile(linearRegressionMLE.train, x, y)
    test_y_linear = metrics_tracker.profile(linearRegressionMLE.predict, x)
    metrics_tracker.mean_squared_error(test_y_linear, y)
    err = metrics_tracker.mean_squared_error(test_y_linear, y)
    print("Squared Mean Error: " + str(err))

    plt.scatter(x, y)
    plt.plot(x, test_y_linear)
    plt.show()

def test_linear_regression_mle_poly():
    #This test shows how MLE is prone to overfitting
    print("Test Linear Regression MLE:  ")
    metrics_tracker = MetricsTracker()
    n_samples = 20
    initial_x = np.linspace(0, 2 * np.pi, n_samples).reshape(n_samples, 1)
    degree_x = 12
    noise = np.random.randn(initial_x.shape[0]).reshape(n_samples, 1)
    for degree_x in range(1, 12, 4):
        x = initial_x

        for i in range(2, degree_x + 1):
            x = np.concatenate((x, pow(initial_x, i)), axis = 1)


        
        y = -4.0*np.sin(initial_x) + noise*0.5
        y = y.reshape((n_samples, 1))


        linearRegressionMLE = LinearRegressionMLE()
        metrics_tracker.profile(linearRegressionMLE.train, x, y)
        test_y_linear = metrics_tracker.profile(linearRegressionMLE.predict, x)
        metrics_tracker.mean_squared_error(test_y_linear, y)
        err = metrics_tracker.mean_squared_error(test_y_linear, y)
        print("Squared Mean Error: " + str(err))

        print (test_y_linear.shape, y.shape, x[:,0].shape)

        
        plt.plot(x[:, 0], test_y_linear, label = str(degree_x))
    plt.scatter(x[:, 0], y)
    plt.legend(loc='best')
    plt.show()


def test_linear_regression_map_simple():
    print("Test Linear Regression MAP:  ")
    metrics_tracker = MetricsTracker()
    n_samples = 200
    x = np.linspace(0, 2 * np.pi, n_samples).reshape(n_samples, 1)
    noise = np.random.randn(x.shape[0]).reshape(n_samples, 1)
    y = -4*x + 7  + noise*2.0
    y = y.reshape(n_samples, 1)

    linearRegressionMAP = LinearRegressionMAP()
    metrics_tracker.profile(linearRegressionMAP.train, x, y, 1.0)
    test_y_linear = metrics_tracker.profile(linearRegressionMAP.predict, x)
    metrics_tracker.mean_squared_error(test_y_linear, y)
    err = metrics_tracker.mean_squared_error(test_y_linear, y)
    print("Squared Mean Error: " + str(err))

    plt.scatter(x, y)
    plt.plot(x, test_y_linear)
    plt.show()

#MAP vs MLE: Check that MAP makes overfitting small
def test_linear_regression_map_mle_poly():
    np.random.seed(5)
    metrics_tracker = MetricsTracker()
    n_samples = 20
    initial_x = np.linspace(0, 2 * np.pi, n_samples).reshape(n_samples, 1)
    degree_x = 10
    noise = np.random.randn(initial_x.shape[0]).reshape(n_samples, 1)
    x = initial_x

    for i in range(2, degree_x + 1):
        x = np.concatenate((x, pow(initial_x, i)), axis = 1)
    
    y = -4.0*np.sin(initial_x) + noise*0.5
    y_noise_free = -4.0*np.sin(initial_x)
    y = y.reshape((n_samples, 1))


    print("Test Linear Regression MLE:  ")
    linearRegressionMLE = LinearRegressionMLE()
    metrics_tracker.profile(linearRegressionMLE.train, x, y)
    test_y_linear_mle = metrics_tracker.profile(linearRegressionMLE.predict, x)
    err_mle = metrics_tracker.mean_squared_error(test_y_linear_mle, y)


    print("Test Linear Regression MAP:  ")
    linearRegressionMAP = LinearRegressionMAP()
    metrics_tracker.profile(linearRegressionMAP.train, x, y, 0.08)
    test_y_linear_map = metrics_tracker.profile(linearRegressionMAP.predict, x)
    metrics_tracker.mean_squared_error(test_y_linear_map, y)
    err_map = metrics_tracker.mean_squared_error(test_y_linear_map, y)


    print("Squared Mean Error (MLE, MAP): " + str((err_mle, err_map)))
    
    plt.plot(x[:, 0], test_y_linear_map, label = 'MAP')
    plt.plot(x[:, 0], test_y_linear_mle, label = 'MLE')
    plt.scatter(x[:, 0], y)
    plt.legend(loc='best')
    plt.title('Polynomial Linear Regression (Degree 9) - MAP vs MLE')
    plt.show()

def test_bayesian_linear_regression():
    np.random.seed(5)
    metrics_tracker = MetricsTracker()
    n_samples = 20
    initial_x = np.linspace(0, 2 * np.pi, n_samples).reshape(n_samples, 1)
    degree_x = 10
    noise = np.random.randn(initial_x.shape[0]).reshape(n_samples, 1)
    x = initial_x

    for i in range(2, degree_x + 1):
        x = np.concatenate((x, pow(initial_x, i)), axis = 1)
    
    y = -4.0*np.sin(initial_x) + noise*0.5
    y_noise_free = -4.0*np.sin(initial_x)
    y = y.reshape((n_samples, 1))


    print("Test Linear Regression Bayesian:  ")
    bayesianLinearRegression = BayesianLinearRegression()
    metrics_tracker.profile(bayesianLinearRegression.train, x, y, 0.5, 
        np.ones(degree_x + 1).reshape((degree_x+1, 1)), np.eye(degree_x + 1))
    test_y_linear_bayesian, mu, var = metrics_tracker.profile(bayesianLinearRegression.predict, x)

    plt.plot(x[:, 0], mu)
    plt.fill_between(x[:, 0], mu - 2*var, mu + 2*var, alpha = 0.4)
    plt.scatter(x[:, 0], y)
    plt.legend(loc='best')
    plt.title('Bayesian Linear Regression (Degree 9)')
    plt.show()

def test_pca():
    data_ingestor = DataIngestor()
    X, y, _, _ = data_ingestor.load_mnist()

    X = X.T


    pca = PCA()

    dimensionality = [1, 10, 100, 500, 784]
    element = [1, 2, 3, 4]
    

    fig, axes = plt.subplots(len(dimensionality), 1, sharey=True)
    plt.gray()
    for i, big_ax in enumerate(axes, start=0):
        big_ax.set_title('PCs = ' + str(dimensionality[i]))
        big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        big_ax._frameon = False
        _, X_tilde = pca.compute_pca(X, dimensionality[i])
        for j in range(len(element)):
            ax = fig.add_subplot(len(dimensionality), len(element), i*len(element) + j + 1)
            ax.imshow(X_tilde.T[element[j]].reshape([28,28]))
            plt.axis('off')
    
    plt.show()

    eigvals, _ = pca.compute_pca(X)

    reconstruction_error = [np.sum(eigvals[i:]) for i in range(len(eigvals))]
    plt.plot(range(len(reconstruction_error)), reconstruction_error)
    plt.axhline(0, color='black')
    plt.title('Average Construction Error')
    plt.show()

main()