import os


def count_samples(directory):
    total = 0
    for root, dirs, files in os.walk(directory):
        total += len(files)
    return total
