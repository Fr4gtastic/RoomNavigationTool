from keras.models import load_model
import cv2
import os
from convert_image import convert_image

directory = r'images_to_predict'    # TODO: create directory, insert some images
res_directory = r'prediction_results'   # TODO: create directory
model_filename = 'model.h5'

if __name__ == '__main__':
    model = load_model(model_filename)

    for filename in os.listdir(directory):
        im = cv2.imread(filename, 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        prediction = model.predict_classes(convert_image(filename))[0]
        if prediction == 1:
            cv2.putText(im, 'Door', (10, 450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(im, 'Window', (10, 450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(res_directory + filename, im)
