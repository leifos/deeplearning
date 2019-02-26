import os
import argparse


def network_one(learning_rate, iterations, batches):

    print("Combination One with learning rate: {} iterations: {} and batch size: {}".format(learning_rate,iterations,batches))


def network_two(learning_rate, iterations, batches):

    print("Combination Two with learning rate: {} iterations: {} and batch size: {}".format(learning_rate,iterations,batches))


def main(combination, learning_rate, iterations, batches, seed):

    # Set Seed
    print("Seed: {}".format(seed))

    if int(combination)==1:
        network_one(learning_rate,iterations, batches)
    if int(combination)==2:
        network_two(learning_rate, iterations, batches)

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
    iterations = check_param_is_numeric("iterations", args.iterations)
    batches = check_param_is_numeric("batches", args.batches)
    seed = check_param_is_numeric("seed", args.seed)

    main(combination, learning_rate, iterations, batches, seed)