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
np.set_printoptions(threshold=np.nan)
from keras.applications.resnet50 import ResNet50
#from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2

from keras.utils.vis_utils import plot_model
from stacked_keras_model import model_generate
from stacked_keras_model import simple_CNN
from stacked_keras_model import mini_XCEPTION
from stacked_keras_model import Xception
from stacked_keras_model import buildNetwork
from stacked_keras_model import conv_3d
from stacked_keras_model import big_XCEPTION
from stacked_keras_model import inceptionv3_based
from stacked_keras_model import tiny_XCEPTION




timesteps = 3
cnn_filters = [16,32,32,64,128]
fc_layers = [512]

nClasses=9
maxPartsTrain = 88
maxPartsTest = 33
maxFrames = 300   

img_dim = 64


'''train_data   = np.load( '/home/yanhong/Downloads/next_step/JPL_First-Person_Interaction activity_video/data/original/train_data.npy' )
train_labels = np.load( '/home/yanhong/Downloads/next_step/JPL_First-Person_Interaction activity_video/data/original/train_labels.npy' )
train_sizes  = np.load( '/home/yanhong/Downloads/next_step/JPL_First-Person_Interaction activity_video/data/original/train_sizes.npy' )
test_data   = np.load( '/home/yanhong/Downloads/next_step/JPL_First-Person_Interaction activity_video/data/original/validation_data.npy' )
test_labels = np.load( '/home/yanhong/Downloads/next_step/JPL_First-Person_Interaction activity_video/data/original/validation_labels.npy' )
test_sizes  = np.load( '/home/yanhong/Downloads/next_step/JPL_First-Person_Interaction activity_video/data/original/validation_sizes.npy' )'''

train_data   = np.load( '../data/fraFred_1/train_data.npy' )
train_labels = np.load( '../data/fraFred_1/train_labels.npy' )
train_sizes  = np.load( '../data/fraFred_1/train_sizes.npy' )
test_data   = np.load( '../data/fraFred_1/validation_data.npy' )
test_labels = np.load( '../data/fraFred_1/validation_labels.npy' )
test_sizes  = np.load( '../data/fraFred_1/validation_sizes.npy' )


print('original_train_data',train_data.shape,train_labels.shape,train_sizes)#(42,70,4096) (42,70,10),[ 39.  44.  38.  26.  41.  43.  41.  29.  35.  29.  41.  35.  20.  32.  43.
                                                                                              #35.  35.  28.  42.  44.  34.  42.  28.  37.  23.  47.  20.  55.  37.  34.                                                                                             #26.  17.  38.  33.  63.  46.  29.  35.  37.  32.  35.  26.]
print('original_test_data',test_data.shape,test_labels.shape,test_sizes)#(20, 70, 4096) (20, 70, 10),[ 32.  37.  36.  32.  23.  24.  25.  42.  34.  38.  34.  30.  23.  34.  30.
                                                                                           #36.33.  25.  32.  37.]

def cleanSet(data, labels, sizes, maxParts, maxFrames, shape):
    data_clean = np.zeros((maxParts, maxFrames, shape))
    labels_clean = np.zeros((maxParts, maxFrames, nClasses))
    sizes_clean = np.zeros((maxParts))
    # Videos will be split into parts
    partId = 0
    frameId_clean = 0
    for videoId in range(data.shape[0]):
        for frameId in range(int(sizes[videoId])-1):
            #type belong to the definite classification
            if labels[videoId,frameId,9] == 0 or labels[videoId,frameId+1,9] == 0:
                #resort accoording to the partId and frameId_clean
                data_clean[partId,frameId_clean] = data[videoId,frameId]
                labels_clean[partId,frameId_clean] = labels[videoId,frameId,:9]
                #frame add
                frameId_clean += 1
            #
            elif frameId_clean > timesteps:
                sizes_clean[partId] = frameId_clean
                partId += 1
                frameId_clean = 0

        if frameId_clean > timesteps:
            sizes_clean[partId] = frameId_clean
    #print('data_clean.shape',data_clean.shape,'labels_clean.shape:',labels_clean.shape,'sizes_clean.shape:',sizes_clean[:])
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
                #transform the data into image format
                stack_frames += [data[videoId, frameId+j].reshape((img_dim,img_dim,1))]
            video_data += [np.array(stack_frames)]
            video_labels += [np.array(labels[videoId, frameId+timesteps-1])]
        new_data += video_data
        new_labels += video_labels
        
    new_labels = np.array(new_labels)
    new_data = np.array(new_data)
    #print('new_data:',new_data.shape,'new_label:',new_labels)

    new_data = np.transpose(new_data, axes=(1,0,2,3,4))
    new_data = list(new_data)
    #print('new_data.len:',len(new_data),'new_labels.len:',len(new_labels))
    return new_data, new_labels
        
def model_generate_based(input_shape):
    model = Sequential()
    model.add(Convolution2D(16, 5, 5, border_mode='valid',
                            input_shape=input_shape))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2), dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))     
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf')) 
    model.add(Convolution2D(32, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf')) 
    model.add(Convolution2D(64, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))     
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))    
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))    
    model.add(Flatten())
    print(model.outputs.shape)
    return model.outputs
def mini_xception_based(img_input,l2_regularization=0.01):
    regularization = l2(l2_regularization)
    #img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                                            use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                                            use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    residual = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    residual = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    print (x.shape)
    return x


def cnn_module(image):
    cnn = Conv2D(filters=cnn_filters[0], kernel_size=(5,5), padding='same',
                  activation='relu', input_shape=(img_dim,img_dim,1))(image)
    cnn = MaxPooling2D(pool_size=(2,2))(cnn)
    for f in cnn_filters[1:]:
        cnn = Conv2D(filters=f, kernel_size=(5,5), padding='same', activation='relu')(cnn)
        cnn = MaxPooling2D(pool_size=(2,2))(cnn)
    cnn = Flatten()(cnn)
    return cnn

def rnn_module(cnn_flat_list):
    rnn = concatenate(cnn_flat_list)
    rnn = Reshape((timesteps, 4*4*128))(rnn)
    rnn = LSTM(units=1024, input_shape=(timesteps, 4*4*128))(rnn)
    return rnn
    
    

def fc_module(fc):
    for layer in fc_layers:
        fc = Dense(layer, activation='relu')(fc)
        fc = Dropout(0.5)(fc)
    fc = Dense(nClasses, activation = 'softmax')(fc)
    return fc

def global_average_pooling(fc):
    #fc = Conv2D(nClasses, (3, 3),padding='same')(fc)
    fc = Dense(nClasses,activation='softmax')(fc)
    fc = GlobalAveragePooling2D()(fc)
    fc = Activation('softmax',name='predictions')(fc)
    return fc




def buildNetwork():
    inputs_list = list()
    cnn_flat_list = list()
    for i in range(timesteps):
        new_input = Input(shape=(img_dim,img_dim,1))
        #concate the timestamp input image
        inputs_list += [new_input]
        #concate the features extracted from cnn   
        cnn_flat_list += [mini_xception_based(new_input)]
    #x = concatenate(cnn_flat_list)
    rnn = rnn_module(cnn_flat_list)
    predictions = fc_module(rnn)
    #predictions = fc_module(x)
    model = Model(inputs=inputs_list, outputs=predictions)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=1e-5),
              metrics=['accuracy'])
    return model

'''def buildNetwork():
    inputs_list = list()
    cnn_flat_list = list()
    for i in range(timesteps):
        new_input = (img_dim,img_dim,1)
        #concate the timestamp input image
        inputs_list += [new_input]
        #concate the features extracted from cnn   
        cnn_flat_list += [model_generate_based(new_input)]
    #x = concatenate(cnn_flat_list)
    rnn = rnn_module(cnn_flat_list)
    predictions = fc_module(rnn)
    #predictions = fc_module(x)
    model = Model(inputs=inputs_list, outputs=predictions)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=1e-5),
              metrics=['accuracy'])
    return model'''
    
    

if __name__ == '__main__':
    logPath = '../hy_results/mini_xception_based_log.txt'
    base_path = '../trained_model/rnn/'
    trained_models_path = base_path + 'recurrent_cnn'+'_mini_xception_based'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}_fraFred_1.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                     save_best_only=True)
    callbacks = [model_checkpoint]
    
    train_data, train_labels, train_sizes, test_data, test_labels, test_sizes = prepareData()
    print('clean_train_data:',train_data.shape,train_labels.shape,train_sizes[:])
    print('clean_test_data:',test_data.shape,test_labels.shape,test_sizes[:])
    model = buildNetwork()
    model.summary()
    #plot_model(model,to_file='model.png',show_shapes=True)
    
    x_train, y_train = format_data(train_data, train_labels, train_sizes)
    x_test, y_test = format_data(test_data, test_labels, test_sizes)
    print('final_train_data:',len(x_train),'train_label:',len(y_train))
    print('final_test_data:',len(x_test),'test_label:',len(y_test))
    
    '''for i in range(1000):
        
        model.fit(x=x_train, y=y_train, batch_size=32, epochs=20, verbose=2,callbacks=callbacks)  
        loss, acc = model.evaluate(x=x_test, y=y_test, batch_size=32, verbose=1)
        print('\n', loss, acc)'''

    model.fit(x=x_train, y=y_train, batch_size=32, epochs=2500, verbose=1,validation_data=(x_test,y_test),callbacks=callbacks) 
    
    
    



