import cv2
import numpy as np
from keras.models import load_model

model_filename = 'model.h5'

if __name__ == '__main__':
    model = load_model(model_filename)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        # display frame by frame
        ret, frame = cap.read()
        if ret:
            font = cv2.FONT_HERSHEY_SIMPLEX

            transformed_frame = np.resize(frame, (32, 32, 3))
            transformed_frame = np.expand_dims(transformed_frame, axis=0)
            prediction = model.predict_classes(transformed_frame)[0]

            if prediction == 1:
                cv2.putText(frame, 'Door', (10, 450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
            elif prediction == 2:
                cv2.putText(frame, 'Window', (10, 450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
