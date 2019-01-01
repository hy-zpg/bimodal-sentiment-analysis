import numpy as np
import keras
from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dense,Dropout,LSTM
from keras.layers.merge import concatenate
from keras.layers.core import Reshape
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
np.set_printoptions(threshold=np.nan)

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



timesteps = 15
cnn_filters = [64,64,128]
fc_layers = [512]

nClasses=9
#not artribute number
#maxPartsTest = 31
#maxFrames = 30
maxPartsTrain = 100
maxPartsTest = 40
maxFrames = 150   
img_dim = 64


train_data   = np.load( '../hy_dataset/train_data.npy' )
train_labels = np.load( '../hy_dataset/train_labels.npy' )
train_sizes  = np.load( '../hy_dataset/train_sizes.npy' )
test_data   = np.load( '../hy_dataset/test_data.npy' )
test_labels = np.load( '../hy_dataset/test_labels.npy' )
test_sizes  = np.load( '../hy_dataset/test_sizes.npy' )

'''train_data   = np.load( '../dataset/train_data.npy' )
train_labels = np.load( '../dataset/train_labels.npy' )
train_sizes  = np.load( '../dataset/train_sizes.npy' )
test_data   = np.load( '../dataset/test_data.npy' )
test_labels = np.load( '../dataset/test_labels.npy' )
test_sizes  = np.load( '../dataset/test_sizes.npy' )'''

print ('train_data:',train_data.shape)
print ('train_labels:',train_labels.shape)
print ('train_sizes:',train_sizes)
print ('test_data:',test_data.shape)
print ('test_labels:',test_labels.shape)
print ('test_sizes:',test_sizes)

#print('original_train_data',train_data.shape,train_labels.shape,train_sizes)#(42,70,4096) (42,70,10),[ 39.  44.  38.  26.  41.  43.  41.  29.  35.  29.  41.  35.  20.  32.  43.
                                                                                              #35.  35.  28.  42.  44.  34.  42.  28.  37.  23.  47.  20.  55.  37.  34.                                                                                             #26.  17.  38.  33.  63.  46.  29.  35.  37.  32.  35.  26.]
#print('original_test_data',test_data.shape,test_labels.shape,test_sizes)#(20, 70, 4096) (20, 70, 10),[ 32.  37.  36.  32.  23.  24.  25.  42.  34.  38.  34.  30.  23.  34.  30.
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
            #print('total frame:',int(sizes[videoId]))
            #type belong to the definite classification
            if labels[videoId,frameId,9] == 0 or labels[videoId,frameId+1,9] == 0:
                #resort accoording to the partId and frameId_clean
                data_clean[partId,frameId_clean] = data[videoId,frameId]
                labels_clean[partId,frameId_clean] = labels[videoId,frameId,:9]
                #frame add
                frameId_clean += 1
                #print('frameId_clean:',frameId_clean)
            #
            elif frameId_clean > timesteps:
                sizes_clean[partId] = frameId_clean
                partId += 1
                frameId_clean = 0
                #print('frame_Id is set as 0 and partId is add 1')
            #print('first recycle--videoId,frameId,partId,frameId_clean:',videoId,frameId,partId,frameId_clean)
            #print('first recycel--data_clean,labels_clean,sizes_clean:',data_clean.shape,labels_clean.shape,sizes_clean[:])
        if frameId_clean > timesteps:
            sizes_clean[partId] = frameId_clean
        #print('second recycle--videoId,frameId,partId,frameId_clean:',videoId,frameId,partId,frameId_clean)
        #print('second recycel--data_clean,labels_clean,sizes_clean:',data_clean.shape,labels_clean.shape,sizes_clean[:])
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
            #print('first recycle--video_data.lehgth,video_labels.length',len(video_data),len(video_labels))
        new_data += video_data
        new_labels += video_labels
        #print('second recycle--')
        
    new_labels = np.array(new_labels)
    new_data = np.array(new_data)
    #print('new_data:',new_data.shape,'new_label:',new_labels)

    new_data = np.transpose(new_data, axes=(1,0,2,3,4))
    new_data = list(new_data)
    #print('new_data.len:',len(new_data),'new_labels.len:',len(new_labels))
    return new_data, new_labels
        
        

train_data, train_labels, train_sizes, test_data, test_labels, test_sizes = prepareData()
x_train, y_train = format_data(train_data, train_labels, train_sizes)
#print('x_train',len(x_train))
#print('y_train',len(y_train))
x_test, y_test = format_data(test_data, test_labels, test_sizes)


