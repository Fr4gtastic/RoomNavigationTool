from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import plot_training
from step_decay import step_decay

train_data_dir = r'data/train'
validation_data_dir = r'data/validation'
tensor_board_dir = r'./logs'
nb_train_samples = 1000
nb_validation_samples = 800
epochs = 500
batch_size = 16
img_width = 150
img_height = 150
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

tensor_board = TensorBoard(log_dir=tensor_board_dir,
                           batch_size=32,
                           write_graph=True,
                           write_grads=True,
                           write_images=True,
                           update_freq='epoch')

model_checkpoint = ModelCheckpoint(filepath=model_filename)

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0.001,
                               patience=10,
                               verbose=1,
                               mode='auto')

learning_rate_scheduler = LearningRateScheduler(step_decay)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[tensor_board, model_checkpoint, learning_rate_scheduler, early_stopping])

model.save(model_filename)

plot_training.plot_accuracy(history)
plot_training.plot_loss(history)

# python -m tensorboard.main --logdir ./logs
# http://localhost:6006
