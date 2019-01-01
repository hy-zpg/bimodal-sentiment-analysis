#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 15:14:59 2017

@author: caetano
"""

import numpy as np
import keras
from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dense,Dropout,LSTM
from keras.layers.merge import concatenate
from keras.layers.core import Reshape
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.applications.inception_v3 import InceptionV3
np.set_printoptions(threshold=np.nan)



timesteps = 3
cnn_filters = [20,40,70,100,130]
fc_layers = [1024,512]

nClasses=9
#maxPartsTrain = 66
#maxPartsTest = 31
maxPartsTrain = 77
maxPartsTest = 20
maxFrames = 30   
img_dim = 64


'''train_data   = np.load( '../data/original/train_data.npy' )
train_labels = np.load( '../data/original/train_labels.npy' )
train_sizes  = np.load( '../data/original/train_sizes.npy' )
test_data   = np.load( '../data/original/test_data.npy' )
test_labels = np.load( '../data/original/test_labels.npy' )
test_sizes  = np.load( '../data/original/test_sizes.npy' )'''

train_data   = np.load( '../data/train_data_50.npy' )
train_labels = np.load( '../data/train_labels_50.npy' )
train_sizes  = np.load( '../data/train_sizes_50.npy' )
test_data   = np.load( '../data/validation_data_12.npy' )
test_labels = np.load( '../data/validation_labels_12.npy' )
test_sizes  = np.load( '../data/validation_sizes_12.npy' )


print('train_data:',train_data.shape,train_labels.shape,train_sizes) #(42, 70, 4096) (42, 70, 10) [ 39.  44.  38.  26.  41.  43.  41.  29.  35.  29.  41.  35.  20.  32.  43.
                                                                     # 35.  35.  28.  42.  44.  34.  42.  28.  37.  23.  47.  20.  55.  37.  34.
                                                                     # 26.  17.  38.  33.  63.  46.  29.  35.  37.  32.  35.  26.]
print('test_data:',test_data.shape,test_labels.shape,test_sizes)     # (20, 70, 4096) (20, 70, 10) [ 32.  37.  36.  32.  23.  24.  25.  42.  34.  38.  34.  30.  23.  34.  30.
                                                                     # 36.  33.  25.  32.  37.]


def cleanSet(data, labels, sizes, maxParts, maxFrames, shape):
    data_clean = np.zeros((maxParts, maxFrames, shape))
    labels_clean = np.zeros((maxParts, maxFrames, nClasses))
    sizes_clean = np.zeros((maxParts))
    # Videos will be split into parts
    partId = 0
    frameId_clean = 0
    for videoId in range(data.shape[0]):
        for frameId in range(int(sizes[videoId])-1):
            if labels[videoId,frameId,9] == 0 or \
                        labels[videoId,frameId+1,9] == 0:
                data_clean[partId,frameId_clean] = data[videoId,frameId]
                labels_clean[partId,frameId_clean] = labels[videoId,frameId,:9]
                frameId_clean += 1
            elif frameId_clean > timesteps:
                sizes_clean[partId] = frameId_clean
                partId += 1
                frameId_clean = 0
        if frameId_clean > timesteps:
            sizes_clean[partId] = frameId_clean
    return (data_clean, labels_clean, sizes_clean)



def prepareData():
    # Train videos
    tr_data_clean, tr_labels_clean, tr_sizes_clean = cleanSet(train_data,
                                                              train_labels,
                                                              train_sizes,
                                                              maxPartsTrain,
                                                              maxFrames,
                                                              train_data.shape[2])
    # Test videos
    ts_data_clean, ts_labels_clean, ts_sizes_clean = cleanSet(test_data,
                                                              test_labels,
                                                              test_sizes,
                                                              maxPartsTest,
                                                              maxFrames,
                                                              test_data.shape[2])
    return tr_data_clean, tr_labels_clean, tr_sizes_clean,\
           ts_data_clean, ts_labels_clean, ts_sizes_clean
           

def format_data(data, labels, sizes):
    new_data = list()
    new_labels = list()
    
    # for each video
    for videoId in range(data.shape[0]):
        video_data = list()
        video_labels = list()
        # for each frame of the video
        for frameId in range(int(sizes[videoId]) - timesteps):
            if np.sum(labels[videoId, frameId]) == 0:
                continue
            stack_frames = list()
            # groups "timesteps" frames
            for j in range(timesteps):
                stack_frames += [data[videoId, frameId+j].reshape((img_dim,img_dim,1))]
            video_data += [np.array(stack_frames)]
            video_labels += [np.array(labels[videoId, frameId+timesteps-1])]
        new_data += video_data
        new_labels += video_labels
        
    new_labels = np.array(new_labels)
    new_data = np.array(new_data)
    new_data = np.transpose(new_data, axes=(1,0,2,3,4))
    new_data = list(new_data)
    return new_data, new_labels
        


# One image only
def cnn_module(image):
    cnn = Conv2D(filters=cnn_filters[0], kernel_size=(5,5), padding='same',
                  activation='relu', input_shape=(img_dim,img_dim,1))(image)
    cnn = MaxPooling2D(pool_size=(2,2))(cnn)
    for f in cnn_filters[1:]:
        cnn = Conv2D(filters=f, kernel_size=(5,5), padding='same', activation='relu')(cnn)
        cnn = MaxPooling2D(pool_size=(2,2))(cnn)
    cnn = Flatten()(cnn)
    return cnn

def xception_based(image):
  base_model = InceptionV3(weights='imagenet', include_top=True)
  cnn = Model(inputs=base_model.input,outputs=base_model.get_layer('avg_pool').output)
  #print(cnn.outputs.shape)


def rnn_module(cnn_flat_list):
    rnn = concatenate(cnn_flat_list)
    #rnn = Reshape((timesteps, 130*2*2))(rnn)
    rnn = Reshape((timesteps, 1000))(rnn)
    rnn = LSTM(units=1024, input_shape=(timesteps, 130*2*2))(rnn)
    return rnn
    
    

def fc_module(fc):
    for layer in fc_layers:
        fc = Dense(layer, activation='relu')(fc)
        fc = Dropout(0.5)(fc)
    fc = Dense(nClasses, activation = 'softmax')(fc)
    return fc


def buildNetwork():
    inputs_list = list()
    cnn_flat_list = list()
    for i in range(timesteps):
        new_input = Input(shape=(img_dim,img_dim,1))
        inputs_list += [new_input]
        #cnn_flat_list += [cnn_module(new_input)]
        cnn_flat_list += [xception_based(new_input)]
    rnn = rnn_module(cnn_flat_list)
    predictions = fc_module(rnn)
    
    model = Model(inputs=inputs_list, outputs=predictions)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=1e-5),
              metrics=['accuracy'])
    return model
    
if __name__ == '__main__':
    logPath = '../hy_results/recurrent_cnn_log2.txt'
    base_path = '../trained_model/RNN/'
    trained_models_path = base_path + 'recurrent_cnn'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                     save_best_only=True)
    callbacks = [model_checkpoint]
    train_data, train_labels, train_sizes, test_data, test_labels, test_sizes = prepareData()
    model = buildNetwork()
    model.summary()
    
    x_train, y_train = format_data(train_data, train_labels, train_sizes)
    x_test, y_test = format_data(test_data, test_labels, test_sizes)    
    '''for i in range(1000):
        model.fit(x=x_train, y=y_train, batch_size=32, epochs=20, verbose=2,callbacks=callbacks)  
        loss, acc = model.evaluate(x=x_test, y=y_test, batch_size=32, verbose=1)
        print('\n', loss, acc)'''
    model.fit(x=x_train, y=y_train, batch_size=32, epochs=2500, verbose=1,validation_data=(x_test,y_test),callbacks=callbacks) 
    
    



