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

'''
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


row_content1 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
row_content2 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
row_content3 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
row_content4 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
row_content5 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
row_content6 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
row_content7 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
row_content8 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
row_content9 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
row_content10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
row_content11 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
for image_file in paths.list_images(LETTER_IMAGES_FOLDER0):
    append_list_as_row('recap.csv', row_content1)

for image_file in paths.list_images(LETTER_IMAGES_FOLDER1):
    append_list_as_row('recap.csv', row_content2)

for image_file in paths.list_images(LETTER_IMAGES_FOLDER2):
    append_list_as_row('recap.csv', row_content3)

for image_file in paths.list_images(LETTER_IMAGES_FOLDER3):
    append_list_as_row('recap.csv', row_content4)

for image_file in paths.list_images(LETTER_IMAGES_FOLDER4):
    append_list_as_row('recap.csv', row_content5)

for image_file in paths.list_images(LETTER_IMAGES_FOLDER5):
    append_list_as_row('recap.csv', row_content6)

for image_file in paths.list_images(LETTER_IMAGES_FOLDER6):
    append_list_as_row('recap.csv', row_content7)

for image_file in paths.list_images(LETTER_IMAGES_FOLDER7):
    append_list_as_row('recap.csv', row_content8)

for image_file in paths.list_images(LETTER_IMAGES_FOLDER8):
    append_list_as_row('recap.csv', row_content9)

for image_file in paths.list_images(LETTER_IMAGES_FOLDER10):
    append_list_as_row('recap.csv', row_content10)

for image_file in paths.list_images(LETTER_IMAGES_FOLDER11):
    append_list_as_row('recap.csv', row_content11)
'''
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


