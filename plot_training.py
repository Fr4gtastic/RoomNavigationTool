import matplotlib.pyplot as pyplot
import numpy


def plot_accuracy(history):
    pyplot.plot(history.history['acc'])
    pyplot.plot(history.history['val_acc'])
    pyplot.title('Model accuracy')
    pyplot.ylabel('Accuracy')
    pyplot.ylabel('Accuracy')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Train', 'Test'], loc='upper left')
    pyplot.savefig('visualization/accuracy.png')
    pyplot.show()


def plot_loss(history):
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('Model loss')
    pyplot.ylabel('Loss')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Train', 'Test'], loc='upper left')
    pyplot.savefig('visualization/loss.png')
    pyplot.show()


def plot_conf_matrix(matrix, total):
    fig, ax = pyplot.subplots()
    ax.matshow(matrix, cmap='binary')
    for (i, j), z in numpy.ndenumerate(matrix):
        ax.text(j, i, '{:0.1f}'.format(z/total), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    pyplot.savefig('visualization/confusion_matrix.png')
    pyplot.show()
