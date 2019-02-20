import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

m,n = housing.data.shape

print(m,n)

housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]

print("Linear Regression Example")

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")

XT = tf.transpose(X)

theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()
    print(theta_value)




from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

print(scaler.fit(housing_data_plus_bias))
print(scaler.mean_)
scaled_housing_data_plus_bias = scaler.transform(housing_data_plus_bias)



print("Gradient Descent Example")



n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")

error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

# cacluated manually using the derivative
#gradients = 2/m * tf.matmul(tf.transpose(X), error)

# calculated using the autodiff feature.. more efficient
#gradients = tf.gradients(mse, [theta])[0]

#training_op = tf.assign(theta, theta - learning_rate * gradients)

# calculated using tfs gradient descent optimizer
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#training_op = optimizer.minimize(mse)

# calculated using tfs momentum descent optimizer - faster


#optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
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
