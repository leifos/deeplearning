import os
import argparse
from ch10_tensorflow_mlp import *

def network_one(learning_rate, epochs, batches):

    print("Perceptron Network with One Hidden Layer")
    print("Combination One with learning rate: {} epochs: {} and batch size: {}".format(learning_rate, epochs, batches))
    mlp_network(1, learning_rate, epochs, batches, activation_func=heavy_side)


def network_two(learning_rate, epochs, batches):

    print("Sigmoid Network with One Hidden Layer")
    print("Combination Two with learning rate: {} epochs: {} and batch size: {}".format(learning_rate, epochs, batches))
    mlp_network(1, learning_rate, epochs, batches, activation_func=tf.nn.sigmoid)

def network_three(learning_rate, epochs, batches):

    print("Perceptron Network with Two Hidden Layers")
    print("Combination Three with learning rate: {} epochs: {} and batch size: {}".format(learning_rate, epochs, batches))
    mlp_network(2, learning_rate, epochs, batches, activation_func=heavy_side)


def network_four(learning_rate, epochs, batches):
    print("Sigmoid Network with Two Hidden Layer")
    print("Combination Four with learning rate: {} epochs: {} and batch size: {}".format(learning_rate, epochs, batches))
    mlp_network(2, learning_rate, epochs, batches, activation_func=tf.nn.sigmoid)



def main(combination, learning_rate, epochs, batches, seed):

    # Set Seed
    print("Seed: {}".format(seed))

    if int(combination)==1:
        network_one(learning_rate, epochs, batches)
    if int(combination)==2:
        network_two(learning_rate, epochs, batches)

    if int(combination)==3:
        network_three(learning_rate, epochs, batches)

    if int(combination)==4:
        network_four(learning_rate, epochs, batches)

    print("Done!")

def check_param_is_numeric(param, value):

    try:
        value = float(value)
    except:
        print("{} must be numeric".format(param))
        quit(1)
    return value


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Assignment Program")
    arg_parser.add_argument("combination", help="Flag to indicate which network to run")
    arg_parser.add_argument("learning_rate", help="Learning Rate parameter")
    arg_parser.add_argument("iterations", help="Number of iterations to perform")
    arg_parser.add_argument("batches", help="Number of batches to use")
    arg_parser.add_argument("seed", help="Seed to initialize the network")

    args = arg_parser.parse_args()

    combination = check_param_is_numeric("combination", args.combination)
    learning_rate = check_param_is_numeric("learning_rate", args.learning_rate)
    epochs = check_param_is_numeric("epochs", args.iterations)
    batches = check_param_is_numeric("batches", args.batches)
    seed = check_param_is_numeric("seed", args.seed)

    main(combination, learning_rate, int(epochs), int(batches), int(seed))
