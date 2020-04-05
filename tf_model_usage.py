import os

import cv2
print("Opencv Version: ", cv2.__version__)
# Pillow modulu de OpenCV gibi, basit bir goruntu isleme modulu
from PIL import Image

import numpy as np
print("Numpy Version: ", np.__version__)
import pandas as pd
print("Pandas Version: ", pd.__version__)

import random

import tensorflow as tf
import tensorflow.keras as keras
print("Tensorflow Version Should be 2.1: ", tf.__version__)
print("Keras Version: ", keras.__version__)

# Bunlar neural network kurulurken kullaniliyor
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dropout, Dense

from tensorflow.keras.utils import to_categorical

# makine ogrenmesinde train ve test datalarini ayiran fonksiyon
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt

# Reading the input images and putting them into a numpy array
data=[]
labels=[]

# image size'leri 30x30 seklinde kucultmek icin kullanacagiz.
height = 30
width = 30
# RGB icin
channels = 3
# sinif sayisi
num_classes = 43


#path = "./gtsrb-dataset/Train/1/"
#a = "00001_00000_00000.png"

path = "./"
a = "yaya-bölgesi-azami-hız-sınırı-20-km.jpg"

try:
    image=cv2.imread(path+a) # siradaki image'i imread ile okuyor.
    image_from_array = Image.fromarray(image, 'RGB')  # ???? https://pillow.readthedocs.io/en/3.1.x/reference/Image.html
    size_image = image_from_array.resize((height, width))
    # image'lar data list'ine numpy array olarak append ediliyor.
    data.append(np.array(size_image))
except Error:
    print("Error! goruntuyu alamadik.")



# Recreate the exact same model, including its weights and the optimizer
loaded_model = tf.keras.models.load_model('model1.h5')

# Show the model architecture
#loaded_model.summary()


X_test = np.array(data)
X_test = X_test.astype('float32')/255  # <class 'numpy.ndarray'>
pred = loaded_model.predict_classes(X_test)

#plt.imshow(image)
#plt.show()
#class_of_prediction = loaded_model.predict(image)[0]

print("Class of Prediction: ", pred)
