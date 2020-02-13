#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals
from tensorflow.keras.datasets import mnist

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def load_dataset_mnist():
    
    (images_train, labels_train), (images_test,labels_test) = mnist.load_data()
    
    return images_train, labels_train, images_test,labels_test

print("\nLoading Dataset:")
images_train, labels_train, images_test,labels_test = load_dataset_mnist()
train_images, test_images = images_train / 255.0, images_test / 255.0
print("\nLoad Dataset Successful:")

num_classes = 10

class CNN_Model(tf.keras.Model):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.layer_1 = tf.keras.layers.Conv2D(filters= 32,kernel_size=(3,3), 
                                              activation = tf.nn.relu)
        self.layer_2 = tf.keras.layers.MaxPooling2D((2,2))
        self.layer_3 = tf.keras.layers.Conv2D(filters = 64, kernel_size=(3,3),
                                              activation = tf.nn.relu)
        self.layer_4 = tf.keras.layers.MaxPooling2D((2,2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(units= 64, activation= tf.nn.relu)
        self.out = tf.keras.layers.Dense(units = num_classes, activation= 'softmax')
    
    def call(self, input_layers):
        cnv = self.layer_1(input_layers)
        cnv = self.layer_2(cnv)
        cvn = self.layer_3(cnv)
        cnv = self.layer_4(cnv)
        den = self.flatten(cnv)
        den = self.dense_1(den)
        return self.out(den)

#train_iamges = tf.data.Dataset.from_tensors(train_images)
#ds = tf.data.Dataset.from_tensor_slices(train_images)
train_images = np.expand_dims(train_images, axis=3) # 4D Req for 2DCNN Layer
print("\nCreating Conv Network Model")
cnn = CNN_Model()
print("\nFeeding Training Data into Model")
#cnn(train_images)


print("\n Compiling Model")
cnn.compile(optimizer=tf.keras.optimizers.Adam(),  # Optimizer
              # Loss function to minimize
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              # List of metrics to monitor
              metrics=['sparse_categorical_accuracy'])

print("\n Fitting Model")
history = cnn.fit(train_images[:54000], labels_train[:54000],
                    batch_size=64,
                    epochs=3,
                    # We pass some validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_data=(train_images[54000:], labels_train[54000:]))

print("\nPlottig History")
print('\nhistory dict:', history.history)


#%% Predictions
n = 3
plt.imshow(test_images[n])
test =np.expand_dims(test_images[n:n+1],axis=3)
print(test.shape)
print(np.round(cnn.predict(test)))
#predictions = cnn.predict(test_images[]:3])
