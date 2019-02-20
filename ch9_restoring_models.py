import tensorflow as tf

"""
https://www.tensorflow.org/guide/saved_model

https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
"""

def main():

    print("Restoring Models Example")

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("tmp/my_final_model.ckpt.meta")
        saver.restore(sess, "tmp/my_final_model.ckpt")
        print(sess.run("theta:0"))

if __name__ == "__main__":
    main()


