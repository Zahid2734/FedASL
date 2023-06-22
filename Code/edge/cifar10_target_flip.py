import numpy as np
import keras
import random
import statistics
import pickle
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras import backend as K
from Code.utils.model import SimpleMLP4

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
nb_classes= 10
y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

loss = 'categorical_crossentropy'
metrics = ['accuracy']
optimizer = 'adam'

smlp_local = SimpleMLP4()
local_model = smlp_local.build()
local_model.compile(loss=loss,
                optimizer=optimizer,
                metrics=metrics)

data6= '../../Data/cifar10/target/target_flipping/FedSTD_loss_0.3_1_0.3_100_200_30_target_flip_cifar10.pkl'
FedSTD_p1= open_file(data6)
# set local model weight to the weight of the global model
local_model.set_weights(FedSTD_p1[5][199])

score = local_model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

Y_test = np.argmax(y_test, axis=1) # Convert one-hot to index
y_pred = np.argmax(local_model.predict(x_test), axis=-1)
print(classification_report(Y_test, y_pred))

matrix = confusion_matrix(y_true=Y_test, y_pred=y_pred)
print(matrix)
print(matrix.diagonal()/matrix.sum(axis=1))