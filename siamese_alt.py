#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import RMSprop

import tensorflow as tf


num_classes = 10
epochs = 10


def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.keras.backend.sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    #Contrastive loss from Hadsell-et-al
    
    margin = 1
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.keras.backend.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    # Only consider the class with the most minimum samples
    # All classes are only considered for n samples
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1 
    
    for d in range(num_classes):
        for i in range(n):
            # Create Same Pair
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]] # Add a new +ve pair
            # Create Different Pair
            inc = random.randrange(1, num_classes) # to choose other class 
            dnew = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dnew][i]
            pairs += [[x[z1], x[z2]]] # Add a new -ve pair
            labels += [1, 0] # Add new labels
    return np.array(pairs), np.array(labels)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return tf.keras.backend.mean(tf.equal(y_true, tf.cast(y_pred < 0.5, y_true.dtype)))


def load_dataset_mnist():
    
    (images_train, labels_train), (images_test,labels_test) = mnist.load_data()
    
    return (images_train, labels_train), (images_test,labels_test)


#%% Create Data
    

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = load_dataset_mnist()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_shape = x_train.shape[1:]

# create training+test positive and negative pairs

#np.where(a < 5, a, 10*a)
# result =array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])

digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
# returns indices where labels are equal to i in range(10)

train_pairs, train_y = create_pairs(x_train, digit_indices)

# tr_pairs has pairs of digits of Trainset

# similarly for test
digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
test_pairs, test_y = create_pairs(x_test, digit_indices)

train_pairs = train_pairs.astype('float64')
train_y = train_y.astype('float64')
test_pairs = test_pairs.astype('float64')
test_y = test_y.astype('float64')


#%%
# network definition
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_y,
          batch_size=128,
          epochs=epochs,
          validation_data=([test_pairs[:, 0], test_pairs[:, 1]], test_y))

#%% Prediction
# compute final accuracy on training and test sets
y_pred = model.predict([train_pairs[:, 0], train_pairs[:, 1]])
train_acc = compute_accuracy(train_y, y_pred)
y_pred = model.predict([test_pairs[:, 0], test_pairs[:, 1]])
test_acc = compute_accuracy(test_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * train_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * test_acc))

# %% Manual Predict

n = 11
m = 11
#temp =model.predict([ train_pairs[0:1,0],train_pairs[0:1,1] ] )
pred_prob = model.predict([ x_train[n:n+1], x_train[m:m+1] ] )

temp = np.round( 1 - pred_prob.ravel() )
print(pred_prob)

res = ['Not Same' , 'Same']
fig,(ax1, ax2) = plt.subplots(1, 2)
title_str = "Image Pair Similarity: "+res[int(temp)]
ax1.imshow(x_train[n])
ax2.imshow(x_train[m])
fig.suptitle(title_str)











