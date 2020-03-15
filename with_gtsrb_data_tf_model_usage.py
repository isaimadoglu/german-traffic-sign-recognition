"""
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception as exception:
  print(exception)
"""

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

# neural network input katmani icin??
n_inputs = height * width*channels

# NOT ALALIM
for i in range(num_classes) :
    path = "./gtsrb-dataset/Train/{0}/".format(i)
    print(path)
    Class=os.listdir(path)

    # For dongusu ile i'inci class'taki fotograflarin uzerinden geciyor.
    for a in Class:
        try:
            image=cv2.imread(path+a) # siradaki image'i imread ile okuyor.
            image_from_array = Image.fromarray(image, 'RGB')  # ???? https://pillow.readthedocs.io/en/3.1.x/reference/Image.html
            size_image = image_from_array.resize((height, width))
            # image'lar data list'ine numpy array olarak append ediliyor.
            data.append(np.array(size_image))
            labels.append(i) # etiketler '0, 1, 2, 3, .. ,42' seklinde
        except AttributeError:
            print("Error! goruntuyu alamadik.")

x_train=np.array(data)
# Her bir piksel 0-255 araliginda deger aliyor ya. Ben bu degerleri 0-1 araligina normalize etmek istersem ne yaparim? 255'e bolerim.
x_train= x_train/255.0

y_train=np.array(labels)
# from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes) # Using one hot encoding



# Recreate the exact same model, including its weights and the optimizer
loaded_model = tf.keras.models.load_model('model1.h5')

# Show the model architecture
loaded_model.summary()


# Split Data
# from sklearn.model_selection import train_test_split
# X'ler veri, Y'ler label
X_train,X_valid,Y_train,Y_valid = train_test_split(x_train,y_train,test_size = 0.3,random_state=0) # X_valid = X_test olarak dusunebiliriz
print("Train :", X_train.shape)
print("Valid :", X_valid.shape)



# Predicting Test data
y_test=pd.read_csv("./gtsrb-dataset/Test.csv")
labels=y_test['Path'].to_numpy()
y_test=y_test['ClassId'].values

data=[]

for f in labels:
    image=cv2.imread('./gtsrb-dataset/Test/'+f.replace('Test/', '')) # ???
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((height, width))
    data.append(np.array(size_image))

X_test=np.array(data)
X_test = X_test.astype('float32')/255
pred = loaded_model.predict_classes(X_test)


# Accuracy with the test data
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)



first_in = 1
second_in = 2
class_of_prediction = loaded_model.predict_classes(X_test[first_in:second_in])[0]

print("Class of Prediction: ", class_of_prediction)
sinif = 1
if(class_of_prediction == sinif):
    print("Birinci Sinif")
