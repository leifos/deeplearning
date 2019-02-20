import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

def main():

    x = tf.Variable(3, name="x")
    y = tf.Variable(4, name="y")

    f = x*x*y + y + 2

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        #x.initializer.run()
        #y.initializer.run()
        results = sess.run(f)
        print(results)


        graph = tf.Graph()
        with graph.as_default():
            x2 = tf.Variable(3)
            print(graph)
            if x2.graph is tf.get_default_graph():
                print("True")

        if x.graph is tf.get_default_graph():
            print("True")



    w = tf.constant(3)
    x = w + 2
    y = x + 5
    z = x * 3

    # the evaluations are performed twice, once for y and z
    with tf.Session() as sess:
        print(y.eval())
        print(z.eval())

    # evaluations are run at the same time
    with tf.Session() as sess:
        yval, zval = sess.run([y,z])
        print(yval)
        print(zval)


if __name__ == "__main__":
    main()
