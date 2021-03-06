import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import fashion_mnist
from helpers import neuron_layer, heavy_side, leaky_relu

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def one_hidden_layers(X, n_hidden1 = 300, n_outputs = 10, activation_func=tf.nn.sigmoid):
    print("Network with one hidden layer")
    with tf.name_scope("dnn"):
        hidden1 = neuron_layer(X, n_hidden1, name="hidden1", activation=activation_func)
        logits = neuron_layer(hidden1, n_outputs, name="outputs")
    return logits

def two_hidden_layers(X, n_hidden1 = 1024, n_hidden2 = 100, n_outputs = 10, activation_func=tf.nn.sigmoid):

    nf1 = 32
    nf2 = 64
    input_layer = tf.reshape(X, [-1, 28, 28, 1])
    print("Network with two hidden layers")
    with tf.name_scope("cnn"):
        conv1 = tf.layers.conv2d( inputs=input_layer, filters=nf1, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=nf2,kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * nf2])

    print("Network with two hidden layers")
    with tf.name_scope("dnn"):
        hidden1 = neuron_layer(pool2_flat, n_hidden1, name="hidden1", activation=activation_func)
        #hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2", activation=activation_func)
        logits = neuron_layer(hidden1, n_outputs, name="outputs")

    return logits


def mlp_network(layers, learning_rate, epochs, batches, activation_func):
    n_inputs = 28*28
    learning_rate = learning_rate
    n_epochs = epochs
    batch_size = batches

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")

    if layers == 1:
        logits = one_hidden_layers(X=X, activation_func=activation_func)
    else:
        logits = two_hidden_layers(X=X, activation_func=activation_func)

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("train"):
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)

        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y ,1)
        accuracy = tf.reduce_mean( tf.cast( correct, tf.float32) )

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    mnist = input_data.read_data_sets("/tmp/data")

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples//batch_size):
                X_batch,y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
                acc_train = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
                acc_val = accuracy.eval(feed_dict={X: mnist.validation.images, y:mnist.validation.labels})

                print(epoch, "Train Accuracy: {:3f}  Validation Accuracy: {:3f}".format(acc_train, acc_val), end="\r")

            save_path = saver.save(sess,"tmp/my_model_final.ckpt")
        print("Train Accuracy: {:3f}  Validation Accuracy: {:3f}".format(acc_train, acc_val), end="\r")

        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y:mnist.test.labels})
        print()
        print("Test Accuracy: {:3f}".format(acc_test))



def main(learning_rate, epochs, batches):

    layers = 1
    print("Relu Network")
    mlp_network(layers, learning_rate, epochs, batches, activation_func=tf.nn.relu)



if __name__ == "__main__":
    main(0.005, 10, 100)
