import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


def load_data():
    housing = fetch_california_housing()
    m,n = housing.data.shape
    housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]

    scaler = StandardScaler()
    print(scaler.fit(housing_data_plus_bias))
    #print(scaler.mean_)
    scaled_housing_data_plus_bias = scaler.transform(housing_data_plus_bias)
    #print(type(scaled_housing_data_plus_bias))
    #print(type(housing.target.reshape(-1, 1)))

    return scaled_housing_data_plus_bias, housing.target.reshape(-1, 1)


def main():

    print("Gradient Descent Example")
    X_Train, Y_Train = load_data()
    m,n = X_Train.data.shape

    n_epochs = 1000
    learning_rate = 0.01

    X = tf.constant(X_Train, dtype=tf.float32, name="X")
    y = tf.constant(Y_Train, dtype=tf.float32, name="y")

    theta = tf.Variable(tf.random_uniform([n, 1], -1.0, 1.0), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")

    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")

    # calculated using tfs gradient descent optimizer
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    # calculated using tfs momentum descent optimizer - faster
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

    training_op = optimizer.minimize(mse)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch ", epoch, " MSE = ", mse.eval())
                sess.run(training_op)
        best_theta = theta.eval()

        print(best_theta)

if __name__ == "__main__":
    main()
