#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.model_selection import train_test_split
#tf.disable_v2_behavior()
from Model import model

def data_train_test():
    y_train = []
    for i in range(1, 6):
        fd_train = open("sample_data/cifar_10_data/data_batch_{}".format(i), 'rb')
        train_dict = pickle.load(fd_train, encoding='latin1')
        fd_train.close()

        if i == 1: 
            x_train = train_dict['data']
        else: 
            x_train = np.vstack((x_train, train_dict['data']))

        y_train = y_train + train_dict['labels']

    x_train = x_train.reshape((len(x_train), 3, 32, 32))
    x_train = np.rollaxis(x_train, 1, 4).astype('float32')
    y_train = np.array(y_train)
    
    fd = open("sample_data/cifar_10_data/test_batch", 'rb')
    dict = pickle.load(fd, encoding='latin1')
    fd.close()
    
    x_test, y_test = dict['data'], dict['labels']
    x_test = x_test.reshape((len(x_test), 3, 32, 32))
    x_test = np.rollaxis(x_test, 1, 4).astype('float32')
    

    return x_train, y_train,x_test,y_test

x_train, y_train, x_test, y_test=data_train_test() # create data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=5000, random_state=42) # creating validation sets
y_train, y_val, y_test = np.eye(10)[y_train], np.eye(10)[y_val],np.eye(10)[np.array(y_test)] 

#x_train /= 255
#x_val /= 255
x_test /= 255

x, y_true, y_pred, train_mode, fully_connected_1, latent_space_flat= model()
saver = tf.train.Saver()
tf_session = tf.Session()
saver.restore(tf_session, 'sample_data/model_save/my-model')
print("model is loaded")


# In[ ]:


# accuracy calculation
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1)), tf.float32))

test_accuracy = tf_session.run(accuracy, {x: x_test, y_true: y_test, train_mode: False})

print("Test set accuracy {:.2f}%".format(test_accuracy*100))
tf_session.close()

