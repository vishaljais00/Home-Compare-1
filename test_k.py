from keras.models import load_model
from keras.preprocessing import image
import numpy as np

model = load_model('first_try_model.h5')

img_pred = image.load_img(r'datasets\image_classification\test_dog_cat\cats\80.jpg', target_size=(150, 150))  # cat
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis=0)

rslt = model.predict(img_pred)
print(rslt[0][0])

if rslt[0][0] == 1:
    prediction = 'dog'

else:
    prediction = 'cat'

print(prediction)
