import matplotlib
import os
import ktrain
from ktrain import vision as vis
from keras.preprocessing.image import image
from keras.metrics import categorical_accuracy
import os.path
from imutils import paths
import numpy as np
import re
import csv
import cv2
import pandas as pd
from PIL import Image
from matplotlib import image
from matplotlib import pyplot
from helpers import resize_to_fit
from sklearn.utils import shuffle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout, GlobalAveragePooling2D,BatchNormalization
import sklearn

train_data = 'Large'
LETTER_IMAGES_FOLDER0 = "Bicycle"
LETTER_IMAGES_FOLDER1 = "Bridge"
LETTER_IMAGES_FOLDER2 = "Bus"
LETTER_IMAGES_FOLDER3 = "Car"
LETTER_IMAGES_FOLDER4 = "Chimney"
LETTER_IMAGES_FOLDER5 = "Crosswalk"
LETTER_IMAGES_FOLDER6 = "Hydrant"
LETTER_IMAGES_FOLDER7 = "Motorcycle"
LETTER_IMAGES_FOLDER8 = "Mountain"
LETTER_IMAGES_FOLDER10 = "Palm"
LETTER_IMAGES_FOLDER11 = "Traffic Light"
from csv import writer


images = []
for image_file in paths.list_images(LETTER_IMAGES_FOLDER0):
    image = cv2.imread(image_file)
    rgbimage = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    imgresize = resize_to_fit(rgbimage, 120, 120)
    images.append(imgresize / 255.0)
for image_file in paths.list_images(LETTER_IMAGES_FOLDER1):
    image = cv2.imread(image_file)
    rgbimage = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    imgresize = resize_to_fit(rgbimage, 120, 120)
    images.append(imgresize / 255.0)
for image_file in paths.list_images(LETTER_IMAGES_FOLDER2):
    image = cv2.imread(image_file)
    rgbimage = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    imgresize = resize_to_fit(rgbimage, 120, 120)
    images.append(imgresize / 255.0)
for image_file in paths.list_images(LETTER_IMAGES_FOLDER3):
    image = cv2.imread(image_file)
    rgbimage = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    imgresize = resize_to_fit(rgbimage, 120, 120)
    images.append(imgresize / 255.0)
for image_file in paths.list_images(LETTER_IMAGES_FOLDER4):
    image = cv2.imread(image_file)
    rgbimage = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    imgresize = resize_to_fit(rgbimage, 120, 120)
    images.append(imgresize / 255.0)
for image_file in paths.list_images(LETTER_IMAGES_FOLDER5):
    image = cv2.imread(image_file)
    rgbimage = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    imgresize = resize_to_fit(rgbimage, 120, 120)
    images.append(imgresize / 255.0)
for image_file in paths.list_images(LETTER_IMAGES_FOLDER6):
    image = cv2.imread(image_file)
    rgbimage = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    imgresize = resize_to_fit(rgbimage, 120, 120)
    images.append(imgresize / 255.0)
for image_file in paths.list_images(LETTER_IMAGES_FOLDER7):
    image = cv2.imread(image_file)
    rgbimage = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    imgresize = resize_to_fit(rgbimage, 120, 120)
    images.append(imgresize / 255.0)
for image_file in paths.list_images(LETTER_IMAGES_FOLDER8):
    image = cv2.imread(image_file)
    rgbimage = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    imgresize = resize_to_fit(rgbimage, 120, 120)
    images.append(imgresize / 255.0)
for image_file in paths.list_images(LETTER_IMAGES_FOLDER10):
    image = cv2.imread(image_file)
    rgbimage = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    imgresize = resize_to_fit(rgbimage, 120, 120)
    images.append(imgresize / 255.0)
for image_file in paths.list_images(LETTER_IMAGES_FOLDER11):
    image = cv2.imread(image_file)
    rgbimage = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    imgresize = resize_to_fit(rgbimage, 120, 120)
    images.append(imgresize / 255.0)

images = np.array(images)
data = pd.read_csv('recap.csv')
print(images.shape)
labels = np.array(data)
print(labels.shape)
images, labels = shuffle(images, labels)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(images, labels, test_size=0.1)

model = Sequential()

model.add(Conv2D(16, (3, 3), activation='relu',))
model.add(BatchNormalization())
model.add(MaxPool2D(2, 2))
model.add(Dropout(.2))

model.add(Conv2D(32, (3, 3), activation='relu',))
model.add(BatchNormalization())
model.add(MaxPool2D(2, 2))
model.add(Dropout(.3))

model.add(Conv2D(64, (3, 3), activation='relu',))
model.add(BatchNormalization())
model.add(MaxPool2D(2, 2))
model.add(Dropout(.4))

model.add(Conv2D(128, (3, 3), activation='relu', ))
model.add(BatchNormalization())
model.add(MaxPool2D(2, 2))
model.add(Dropout(.5))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(.5))

model.add(Dense(11, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=4, validation_data=(X_test, y_test))


