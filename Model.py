#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def model():
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_true = tf.placeholder(tf.float32, shape=[None, 10])
    train_mode = tf.placeholder(tf.bool)
    
    conv1 = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3], padding='same')
    conv1_bn = tf.layers.batch_normalization(conv1,momentum=0.9, training=train_mode)
    conv1_act = tf.nn.relu(conv1_bn)
    conv1_pool = tf.layers.max_pooling2d(conv1_act, pool_size=[2, 2], strides=2, padding='same')
    #conv1_bn = tf.layers.batch_normalization(conv1_pool)
    print("Convolution 1: ", conv1.shape)
    print("Activation Shape: ", conv1_act.shape)
    print("Pooling Shape: ", conv1_pool.shape)
    print("Batch Normalization Shape: ", conv1_bn.shape)
    # 
    conv2 = tf.layers.conv2d(conv1_pool,filters=64,kernel_size=[3, 3], padding='same')
    conv2_bn = tf.layers.batch_normalization(conv2, momentum=0.9, training=train_mode)
    conv2_act = tf.nn.relu(conv2_bn)
    conv2_pool = tf.layers.max_pooling2d(conv2_act, pool_size=[2, 2], strides=2, padding='same')    
    #conv2_bn = tf.layers.batch_normalization(conv2_pool)
    print("Convolution 2: ", conv2.shape)
    print("Activation Shape: ", conv2_act.shape)
    print("Pooling Shape: ", conv2_pool.shape)
    print("Batch Normalization Shape: ", conv2_bn.shape)
    #
    conv3 = tf.layers.conv2d(conv2_pool, filters=128, kernel_size=[3, 3], padding='same')
    conv3_bn = tf.layers.batch_normalization(conv3, momentum=0.9, training=train_mode)
    conv3_act = tf.nn.relu(conv3_bn)
    conv3_pool = tf.layers.max_pooling2d(conv3_act, pool_size=[2, 2], strides=2, padding='same')  
    #conv3_bn = tf.layers.batch_normalization(conv3_pool)
    print("Convolution 3: ", conv3.shape)
    print("Activation Shape: ", conv3_act.shape)
    print("Pooling Shape: ", conv3_pool.shape)
    print("Batch Normalization Shape: ", conv3_bn.shape)
    #
    flat = tf.reshape(conv3_pool, [-1, 4 * 4 * 128])
    print("flatten: ", flat.shape)

    full_connected_1 = tf.layers.dense(flat, units=512)
    activation_4 = tf.nn.relu(full_connected_1)
    dropout_4 = tf.layers.dropout(activation_4, rate=0.5, training=train_mode)
    print("full_connected_1: ", full_connected_1.shape)
    print("activation_4: ", activation_4.shape)
    print("dropout_4: ", dropout_4.shape)

    y_pred = tf.layers.dense(dropout_4, units=10)
    print("y_pred: ", y_pred.shape)
    return x, y_true, y_pred, train_mode, dropout_4, flat

