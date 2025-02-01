## Code for Linear Regression from Scratch

import numpy as np
import matplotlib.pyplot as plt

def linear_regression(X,y):

    ## adding a column of one to X for the bias term b
    X_b = np.c_[np.ones((X.shape[0],1)),X]

    ## calculating the optimal weights using the Normal Equation theta = (X^T * X)^-1 * X^T * y
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    return theta_best

def predict(X,theta):
    ## adding a column of one to X for the bias term b
    X_b = np.c_[np.ones((X.shape[0],1)),X]

    ## calculating the predictions using y = X * theta
    y_pred = X_b.dot(theta)

    return y_pred
def plot(X,y,theta):
    plt.scatter(X,y)
    plt.plot(X,predict(X,theta),color='red')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.show()
if __name__ == '__main__':
    ## generating some random data
    X = 2 * np.random.rand(100,1)
    y = 4 + 3 * X + np.random.randn(100,1)

    ## training the model
    theta_best = linear_regression(X,y)

    ## plotting the results
    plot(X,y,theta_best)