import math


def step_decay(epoch):
    initial_learning_rate = 0.01
    drop = 0.75
    epochs_drop = 5.0
    lrate = initial_learning_rate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
