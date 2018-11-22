from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from pathlib import Path
import plot_training

train_data_dir = r'data/train'
validation_data_dir = r'data/validation'
nb_train_samples = 1000
nb_validation_samples = 800
epochs = 50
batch_size = 16
img_width = 150
img_height = 150
model_filename = 'model.json'
weights_filename = 'weights.h5'

with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)

if Path(weights_filename).exists():
    model.load_weights(weights_filename)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_data_generator = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rescale=1. / 255,
    zca_whitening=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_data_generator = ImageDataGenerator(rescale=1. / 255)

train_generator = train_data_generator.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    save_to_dir=r'data/processed/train')

validation_generator = test_data_generator.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    save_to_dir=r'data/processed/validation')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[reduce_lr])

model.save_weights(weights_filename)

plot_training.plot_accuracy(history)
plot_training.plot_loss(history)
