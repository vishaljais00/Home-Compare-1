# generates multiple data from one image through various opeartions
from keras.preprocessing.image import ImageDataGenerator

# to create a NN model
from keras.models import Sequential

# wil get the CNN, and pooling reduce the size of the data
from keras.layers import Conv2D, MaxPooling2D

# activation functions, dropout to reduce overfitting, flatten converts 2d array to 1d array, dense for layers
from keras.layers import Activation, Dropout, Flatten, Dense

# backend helps to understand the channel
from keras import backend as K

# to perform mathematical opeartions
import numpy as np

# to get the images from our directory
from keras.preprocessing import image

import os

DIR = r'C:\Users\HARSH\Desktop\Hackathon\datasets\image_classification'

# Dimention of our images
img_width, img_height = 150, 150

# Locating the train and test samples

test_data_dir = os.path.join(DIR, 'test_dog_cat')
train_data_dir = os.path.join(DIR, 'train_dog_cat')
nb_train_samples = 1000
nb_validation_samples = 100
epochs = 50     # number of times batches will be sent to the CNN
batch_size = 20     # number of data in each batch

if K.image_data_format() == 'channels_first':   # if rgb
    input_shape = (3, img_width, img_height)

else:
    input_shape = (img_width, img_height, 3)    # 150, 150, 3

# from this we will get 4 images from one images by performing various operations

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# this is for test data, we wont modify it a lot

test_datagen = ImageDataGenerator(rescale=1./255)

# this will generate the train data that has to be given to the Neural Network

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# this will generate the test that that has to be given the Neural Network

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Now we will make a Neural Network using keras

model = Sequential()    # initialisation of the NN

model.add(Conv2D(32, (3, 3), input_shape=input_shape))  # make a CNN and extract 32 features, using 2*2 search matrix

model.add(Activation('relu'))   # activation function is relu

model.add(MaxPooling2D(pool_size=(2, 2)))   # reduce the size without loosing the features

model.summary()

# We will repeat the above steps  2 more times

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())    # converting 2d array to 1d array
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

# Now defining the data that we have to feed into the Neural Netwok
# this will start our neural network

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=nb_validation_samples // batch_size
)

# Will save all the weights for future use

model.save('first_try_model.h5')

img_pred = image.load_img(r'datasets\image_classification\test_dog_cat\dogs\3.jpg', target_size=(150, 150))  # dog
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis=0)

# Now will do the prediction of our image

# rslt = model.predict(img_pred)
# print(rslt[0][0])
#
# if rslt[0][0] == 1:
#     prediction = 'dog'
#
# else:
#     prediction = 'cat'
#
# print(prediction)
