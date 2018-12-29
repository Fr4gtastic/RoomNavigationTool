from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import plot_training
import numpy as np
from step_decay import step_decay
from sklearn.metrics import confusion_matrix, classification_report
from samples_counter import count_samples

train_data_dir = r'data/train'
validation_data_dir = r'data/validation'
tensor_board_dir = r'./logs'
nb_train_samples = count_samples(train_data_dir)
nb_validation_samples = count_samples(validation_data_dir)
epochs = 100
batch_size = 16
img_width = 32
img_height = 32
model_filename = 'model.h5'

model = load_model(model_filename)

train_data_generator = ImageDataGenerator(
    rescale=1. / 255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_data_generator = ImageDataGenerator(rescale=1. / 255)

train_generator = train_data_generator.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_data_generator.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                              patience=10, min_lr=1e-8)
# reduce learning rate by [factor] if [monitor] hasn't changed after [patience] epochs until it reaches the value of
# [min_lr]

tensor_board = TensorBoard(log_dir=tensor_board_dir,
                           batch_size=batch_size,
                           write_graph=True,
                           write_grads=True,
                           write_images=True,
                           update_freq='epoch')
# visualization callback; to see the results use the command
# python -m tensorboard.main --logdir ./logs
# and go to
# http://localhost:6006

model_checkpoint = ModelCheckpoint(filepath=model_filename)
# save the model after each epoch

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0.001,
                               patience=10,
                               verbose=1,
                               mode='auto')
# stop training if [monitor] hasn't changed after [patience] epochs by at least [min_delta]

learning_rate_scheduler = LearningRateScheduler(step_decay)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[tensor_board, model_checkpoint, reduce_lr])

model.save(model_filename)

plot_training.plot_accuracy(history)
plot_training.plot_loss(history)


validation_predict = model.predict_generator(validation_generator, nb_validation_samples // batch_size + 1)
# make predictions for validation set
validation_predict_y_max = np.argmax(validation_predict, axis=1)
print('Confusion matrix')
confusion_matrix = confusion_matrix(validation_generator.classes, validation_predict_y_max)
print(confusion_matrix)
print('\n')
print('Classification report')
target_names = validation_generator.class_indices
print(classification_report(validation_generator.classes, validation_predict_y_max, target_names=target_names))
plot_training.plot_conf_matrix(confusion_matrix, nb_validation_samples)
# plot the confusion matrix
