import tensorflow as tf
import numpy as np
from datetime import datetime
from ch9_gradient_descent import load_data
from ch9_mini_batch_gradient_descent import fetch_batch


def main():
    print("Logging Example")
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_dir = "logs/run-{}".format(now)

    X_train, y_train = load_data()
    m,n = X_train.data.shape
    batch_size = 2500
    n_batches = int(np.ceil(m / batch_size))
    n_epochs = 400
    learning_rate = 0.01

    X = tf.placeholder(dtype=tf.float32, shape=(None, n), name="X")
    y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="y")

    theta = tf.Variable(tf.random_uniform([n, 1], -1.0, 1.0), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")

    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    training_op = optimizer.minimize(mse)


    #Added in logging variables after the construction phase
    # first create a node in the graph that writes the MSE to a TensorBoard Compatiable binary
    mse_summary = tf.summary.scalar("MSE", mse)
    # sets up the writer which will we call to write the values
    file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

    init = tf.global_variables_initializer()
    #Create the Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(X_train, y_train, batch_index, batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

            # Log out the summary string containing the MSE values
            #if batch_index % 5 ==0:
            # In practice - only log every n rounds - otherwise too slow
            summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
            step = epoch * n_batches + batch_index
            file_writer.add_summary(summary_str, step)

            if epoch % 10 == 0:
                print("Epoch ", epoch, " MSE = ", mse.eval(feed_dict={X: X_batch, y: y_batch}))
                # Call the save function to save the model
                # save at intermediate points

                save_path = saver.save(sess,"tmp/my_model.ckpt")
        best_theta = theta.eval()
        print(best_theta)
        # save the final model
        save_path = saver.save(sess,"tmp/my_final_model.ckpt")

    file_writer.close()


if __name__ == "__main__":
    main()
