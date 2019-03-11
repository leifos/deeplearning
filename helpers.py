import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def show_images(X,y):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i], cmap=plt.cm.binary)
        plt.xlabel(y[i])
    plt.show()



def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2.0 / np.sqrt(n_inputs + n_neurons)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]),name="bias")
        Z = tf.matmul(X,W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z


def heavy_side(z, name=None):
    return tf.nn.relu(tf.math.sign(z), name=name)

def leaky_relu(z, name=None):
    return tf.maximum(0.2*z,z, name=name)
