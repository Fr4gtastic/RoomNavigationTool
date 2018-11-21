import matplotlib.pyplot as pyplot


def plot_accuracy(history):
    pyplot.plot(history.history['acc'])
    pyplot.plot(history.history['val_acc'])
    pyplot.title('Model accuracy')
    pyplot.ylabel('Accuracy')
    pyplot.ylabel('Accuracy')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Train', 'Test'], loc='upper left')
    pyplot.show()
    pyplot.savefig('visualization/accuracy.png')


def plot_loss(history):
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('Model loss')
    pyplot.ylabel('Loss')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Train', 'Test'], loc='upper left')
    pyplot.show()
    pyplot.savefig('visualization/loss.png')
