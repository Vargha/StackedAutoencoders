# -*- coding: utf-8 -*-
#####################################################################################
# Team Members:   Vargha Hokmran, Tiffany Tawfic                                    #
# Professor:      Mr. Hemanth Venkateswara                                          #
# Class:          Intro to Deep Learning in Visual Computing                        #
#                 (by Arizona State University)                                     #
#                                                                                   #
# Project Title:  Stacked Autoencoder                                               #
# Description:    Build a stacked autoencoder using layer-by-layer training of      #
#                 autoencoders in an unsupervised manner using the CIFAR-10 dataset #
#####################################################################################

import torchvision.transforms as transforms
import torchvision
import torch

from keras.datasets import cifar10
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import np_utils 

import numpy as np
import matplotlib.pyplot as plt

nr_train = 10000
nr_test = 5000
nr_epochs = 50
num_classes = 10

def rgb_to_grayscale(data, dtype='float32'):
  # https://www.tutorialspoint.com/dip/grayscale_to_rgb_conversion.htm
  r, g, b = np.asarray(.3, dtype=dtype), np.asarray(.59, dtype=dtype), np.asarray(.11, dtype=dtype)
  gs_data = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
  gs_data = np.expand_dims(gs_data, axis=3)
  gs_data = gs_data.reshape(len(gs_data), 32, 32)
  return gs_data

# https://blog.keras.io/building-autoencoders-in-keras.html

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.                      #normalize data
x_test = x_test.astype('float32') / 255.
x_train = x_train[:nr_train]
x_test = x_test[:nr_test]
x_train = rgb_to_grayscale(x_train)
x_test = rgb_to_grayscale(x_test)
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

encoding_dim = 500                                               # this is the size of our encoded representations
input_img = Input(shape=(32*32,))

x = Dense(32*32, activation='relu')(input_img)
encoded1 = Dense(1000, activation='relu')(x)
encoded2 = Dense(800, activation='relu')(encoded1)

y = Dense(encoding_dim, activation='relu')(encoded2)

decoded2 = Dense(800, activation='relu')(y)
decoded1 = Dense(1000, activation='relu')(decoded2)
z = Dense(32*32, activation='sigmoid')(decoded1)            # "decoded" is the lossy reconstruction of the input
autoencoder = Model(input_img, z)
encoder = Model(input_img, y)                              #encoder is the model of the autoencoder slice in the middle

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.fit(x_train, x_train,
                 epochs=nr_epochs,
                 batch_size=256,
                 shuffle=True,
                 validation_data=(x_test, x_test))

# For 3-layer encoders and decoders, you have to call all 3 layers for defining decoder.
# Retrieve the last 3 layer of the autoencoder model
# https://stackoverflow.com/questions/44472693/how-to-decode-encoded-data-from-deep-autoencoder-in-keras-unclarity-in-tutorial
encoded_input = Input(shape=(encoding_dim,))
deco = autoencoder.layers[-3](encoded_input)
deco = autoencoder.layers[-2](deco)
deco = autoencoder.layers[-1](deco)
decoder = Model(encoded_input, deco)

# encode and decode some inputs
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
predicted = autoencoder.predict(x_test)


n = 10  # how many photos to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# ---------------------------------------------------------------
# https://github.com/rjpg/bftensor/blob/master/Autoencoder/src/AutoEncoderMNIST.py
# To lock the weights of the encoder on post-training 
for layer in encoder.layers : layer.trainable = False

y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels

# Define new model encoder -> Dense  10 neurons with soft max for classification 
num_classes = 10
out2 = Dense(num_classes, activation='softmax')(encoder.output)

# Fine tuning with 10-labeled samples
fine_tune_size = 10
x_train10 = x_train[:fine_tune_size]
x_test10 = x_test[:fine_tune_size]
y_train10 = y_train[:fine_tune_size]
y_test10 = y_test[:fine_tune_size]

newmodel = Model(encoder.input, out2)
newmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

newmodel.fit(x_train10, y_train10,
      epochs=50,
      batch_size=256,
      shuffle=True,
      validation_data=(x_test10, y_test10))
scoreFor10 = newmodel.evaluate(x_test10, y_test10, verbose=1)   

# Fine tuning with 100-labeled samples
fine_tune_size = 100
x_train100 = x_train[:fine_tune_size]
x_test100 = x_test[:fine_tune_size]
y_train100 = y_train[:fine_tune_size]
y_test100 = y_test[:fine_tune_size]

newmodel2 = Model(encoder.input, out2)
newmodel2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
newmodel2.fit(x_train100, y_train100,
      epochs=50,
      batch_size=256,
      shuffle=True,
      validation_data=(x_test100, y_test100))

scoreFor100 = newmodel2.evaluate(x_test100, y_test100, verbose=1) 

print("Accuracy with 10-labeled samples: ", scoreFor10[1])
print("Accuracy with 100-labeled samples: ", scoreFor100[1])
