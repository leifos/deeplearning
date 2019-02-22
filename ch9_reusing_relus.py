import os
from datetime import datetime
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def make_relu(X, threshold):
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]),1)
        w = tf.Variable(tf.random_normal(w_shape), name="weight")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X,w),b, name="z")
        return tf.maximum(z, 0.0, name="relu")


def main():

    print("Re-Using RELUs example")

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_dir = "logs/run-{}".format(now)

    n_features = 3
    threshold = tf.Variable(0.0, name="threshold")
    X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
    relus = []
    for i in range(5):
        relus.append(make_relu(X, threshold))

    output = tf.add_n(relus, name="output")

    init = tf.global_variables_initializer()

    file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

    with tf.Session() as sess:
        init.run()
        graph = tf.Graph()

    file_writer.close()

    print("Load Tensorboard and inspect the graph")

if __name__ == "__main__":
    main()
