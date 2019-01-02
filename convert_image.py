from keras.preprocessing import image
import numpy as np


def convert_image(image_name, target_size=(32, 32)):
    img = image.load_img(image_name, target_size=target_size)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    return img_tensor
