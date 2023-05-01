#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.manifold import TSNE

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from Model import model


# In[ ]:


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


# In[ ]:


def plot_loss_acc(train_acc,train_loss,val_acc,val_loss):
    epochs=[i for i in range(len(train_acc))]
    df_acc = pd.DataFrame({"Train Accuracy":train_acc,"Validation Accuracy":val_acc})
    df_loss = pd.DataFrame({"Train Loss":train_loss,"Validation Loss":val_loss})
    plot_acc = df_acc.plot(xlabel="Epochs",ylabel="Train and Validation Accuracy(%)").get_figure()
    plot_acc.savefig('sample_data/Plots/Train_and_Validation.png')

    plot_loss = df_loss.plot(xlabel="Epochs",ylabel="Train and Validation Loss(%)").get_figure()
    plot_loss.savefig('sample_data/Plots/Train_and_Validation_Loss.png')
 
    return plot_acc,plot_loss

def plot_tsne(latent, labels, file_name):
    tsne_data = TSNE(n_components=2).fit_transform(latent[0])  # (45000, 2) 2d tsne data 
    tsne_data = np.vstack((tsne_data.T, labels)).T
    tsne_df = pd.DataFrame(data=tsne_data, columns=("Dimension 1", "Dimension 2", "label"))
    seaborn.FacetGrid(tsne_df, hue="label", height=6).map(plt.scatter, "Dimension 1", "Dimension 2").add_legend()

    plt.savefig(file_name + '.png')
    print("-> tsne.png saved.")
    plt.show()


# In[ ]:


x_train, y_train, x_test, y_test=data_train_test() # create data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=5000, random_state=42) # creating validation sets
y_train, y_val, y_test = np.eye(10)[y_train], np.eye(10)[y_val],np.eye(10)[np.array(y_test)] # one-hot encoding


# In[ ]:


#x_train.shape,y_train.shape,x_test.shape,y_test.shape,x_val.shape,y_val.shape


# In[ ]:


# we normalize training, validation and test data (0-1 range)
x_train /= 255
x_val /= 255
x_test /= 255


# In[ ]:


data_augment = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
data_augment.fit(x_train)


# In[ ]:


max_epochs = 30
Batch_size = 32
learning_rate = 0.01
tsne_plot = False
early_stoping = False


# In[ ]:


# model
x, y_true, y_pred, train_mode, fully_connected_1, latent_space_flat = model()

# loss function
loss = tf.losses.softmax_cross_entropy(y_true, y_pred)

# accuracy calculation
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1)), tf.float32))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batchnorm

# optimization functions
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

train_optimizer = tf.group([train_optimizer, update_ops]) # for batchnorm

print("-> Max number of epochs = ", max_epochs)
print("-> Size of batch = ", Batch_size)
print("-> Learning_rate = ", learning_rate)
print("-" * 100)


# In[ ]:


tf_session = tf.Session()
tf_session.run(tf.global_variables_initializer())
print("------Session Started------")
train_acc,train_loss,val_acc,val_loss = [],[],[],[] # To plot loss and accurcies, train and validation
#Two parameters defined for early stopping
key_stop=0
key_loss=0
stop_training=False
# Training start here
for epoch in range(max_epochs):
    
    if stop_training == True:
        
        print("-----Early stopping is active.-------")
        print("-----Training Stopped-----")
        
        break
        
    # for each epochs, use batch calculate epoch loss and accuracies to plot.
    
    number_of_batch = 0
    epoch_acc, epoch_loss, epoch_val_acc, epoch_val_loss = 0, 0, 0, 0
    
    for batch_x, batch_y in data_augment.flow(x_train, y_train, batch_size=Batch_size):
        
        _, batch_loss = tf_session.run([train_optimizer, loss],
                                 feed_dict={x: batch_x, y_true:batch_y, train_mode: True})
        
        batch_acc = tf_session.run(accuracy, feed_dict={x: batch_x, y_true:batch_y, train_mode: False})
        
        epoch_loss += batch_loss
        epoch_acc += batch_acc
        number_of_batch += 1
        
        if number_of_batch >= len(x_train)/Batch_size: # stop training as reach the last batch of data
            break
            
        
        
    epoch_loss /= number_of_batch #loss for epoch
    epoch_acc /= number_of_batch # accuracy for epoch

    epoch_val_loss, epoch_val_acc = tf_session.run([loss, accuracy],
                                                                {x: x_val, y_true: y_val,
                                                                 train_mode: False})
        
    epoch_acc *= 100
    epoch_val_acc *= 100
        
    val_loss.append(epoch_val_loss)
    val_acc.append(epoch_val_acc)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
        
    print(
        "Epoch: {} Train Loss: {:.2f} - Validation Loss: {:.2f} | Train Accuracy: {:.2f}% - Validation Accuracy: {:.2f}% ".format(
            epoch + 1, epoch_loss, epoch_val_loss, epoch_acc, epoch_val_acc))
        
    # Early stopping:
    # After 5 epoch if, val loss decrease 3 times consecutive and training keep increasing, then stop, check validation losses
    #key_loss=0
    #key_stop=0
   
    if early_stoping ==True:    
      if len(train_loss)>5:
        if (val_loss[len(train_loss)-1]>val_loss[len(train_loss)-2] and train_loss[len(train_loss)-1]<train_loss[len(train_loss)-2]):
          print("deneme")
          key_stop +=1
          print("key_stop",key_stop)
        else:
          key_stop=0
          print("key_stop",key_stop)


      if key_stop ==3:
        print("key_stop",key_stop)
        stop_training=True

        print("Early stopping is True. Validation losses start to increse and Training loss keeping decrease.")
    #Tsne plotting                
    if tsne_plot==True and (epoch == 3 or epoch == 13 or epoch == 27):
        latent_fc1, latent_flatten = tf_session.run([fully_connected_1, latent_space_flat],
                                              {x: x_test, y_true: y_test, train_mode: False})
        plot_tsne(np.array([latent_fc1]), np.array([np.where(r == 1)[0][0] for r in np.array(y_test)]),
                  "fc1_test_tsne_" + str(epoch))
        plot_tsne(np.array([latent_flatten]), np.array([np.where(r == 1)[0][0] for r in np.array(y_test)]),
                  "flatten_test_tsne_" + str(epoch))


print("Final Epoch: {} Train Loss: {:.2f} - Validation Loss: {:.2f} | Train Accuracy: {:.2f}% - Validation Accuracy: {:.2f}% ".format(
            max_epochs, train_loss[len(train_loss)-1],val_loss[len(val_loss)-1], train_acc[len(train_acc)-1], val_acc[len(val_acc)-1]))
        
plot_loss_acc(train_acc,train_loss,val_acc,val_loss)
        


# In[ ]:


# save model
saver = tf.train.Saver()
#saved_path = saver.save(tf_session, 'trained_model')
#saved_path = saver.save(tf_session, './my-model')
saved_path = saver.save(tf_session,"sample_data/model_save/my-model")
print("Model is saved.")

tf_session.close()

print("Tf Session is closed.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




