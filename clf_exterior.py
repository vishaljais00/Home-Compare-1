from keras.models import load_model
from keras.preprocessing import image
import numpy as np

model = load_model('second_model_exterior.h5')


def pred_exterior(img):
    img_pred = image.load_img(img, target_size=(150, 150))
    img_pred = image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis=0)

    rslt = model.predict(img_pred)

    if rslt[0][0] == 1:
        return 1

    else:
        return 0
