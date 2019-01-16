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

train_data_dir = r'data4/train'
validation_data_dir = r'data/validation'
tensor_board_dir = r'./logs'
validation_split_factor = 0.2
nb_train_samples = count_samples(train_data_dir)
nb_validation_samples = int(nb_train_samples * validation_split_factor)
epochs = 200
batch_size = 32
img_width = 64
img_height = 64
model_filename = 'model.h5'

model = load_model(model_filename)

train_data_generator = ImageDataGenerator(
    rescale=1. / 255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=validation_split_factor)

test_data_generator = ImageDataGenerator(rescale=1. / 255)

train_generator = train_data_generator.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = train_data_generator.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1,
                              patience=5, min_lr=0, min_delta=1e-3)
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
                               min_delta=1e-3,
                               patience=10,
                               verbose=1,
                               mode='auto',
                               restore_best_weights=True)
# stop training if [monitor] hasn't changed after [patience] epochs by at least [min_delta]

learning_rate_scheduler = LearningRateScheduler(step_decay)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=5*nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=5*nb_validation_samples // batch_size,
    callbacks=[tensor_board, model_checkpoint, learning_rate_scheduler])

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
