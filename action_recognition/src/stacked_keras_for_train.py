import keras
import numpy as np
from keras import layers
import pickle 
import scipy.misc
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Activation, Convolution2D, Dropout, Conv2D, Dense
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras.regularizers import l2
from keras.optimizers import Adadelta
from keras.utils import np_utils
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
  
from keras.utils.vis_utils import plot_model 
from stacked_keras_model import model_generate
from stacked_keras_model import simple_CNN
from stacked_keras_model import mini_XCEPTION
from stacked_keras_model import Xception
from stacked_keras_model import buildNetwork
from stacked_keras_model import conv_3d
from stacked_keras_model import big_XCEPTION
from stacked_keras_model import inceptionv3_based;
from stacked_keras_model import tiny_XCEPTION

'''stackLen = 3
maxPartsTrain = 66
maxPartsTest = 31
maxFrames = 30 '''
#the stacklen decide the maxparttrain,maxparttest,maxframe
#stackLen = 3
stackLen = 10
maxPartsTrain = 88#obtained from clean_train_size 84
#second recycle--videoId,frameId,partId,frameId_clean: 41 24 66 0
maxPartsTest = 33#obtained from clean_test_size13
#second recycle--videoId,frameId,partId,frameId_clean: 19 35 30 19
#maxFrames = 30
maxFrames = 300
#first recycle--videoId,frameId,partId,frameId_clean: 10 33 17 22  

#train_50:77 20 [ 0.21052632  0.08571429  0.15384615  0.35833333  0.17391304  0.33802817 0.33333333  0.37254902  0.41176471]
#train_55:84 13 [ 0.16949153  0.2         0.11111111  0.17266187  0.10958904  0.15853659 0.06024096  0.04477612  0.152     ]
#train_52:83 14 [ 0.104       0.04587156  0.23287671  0.15602837  0.17391304  0.30136986 0.23943662  0.22807018  0.22033898]
#train_55_1:86 11 [ 0.12195122  0.11764706  0.11111111  0.14788732  0.10958904  0.14457831 0.12820513  0.12903226  0.13385827]
#train_50_1:79 18 [ 0.23214286  0.23913043  0.23287671  0.23484848  0.265625    0.23376623 0.25714286  0.27272727  0.24137931]
#train_standard:78 19 [ 0.22123894  0.2         0.2         0.24427481  0.20895522  0.23376623 0.20547945  0.22807018  0.22033898]
def cleanSet(data, labels, sizes, maxParts, maxFrames, shape, nClasses=9):
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
            elif frameId_clean > stackLen:
                sizes_clean[partId] = frameId_clean
                partId += 1
                frameId_clean = 0
        if frameId_clean > stackLen:
            sizes_clean[partId] = frameId_clean
            
    return (data_clean, labels_clean, sizes_clean)



'''def prepareData():
    train_data   = np.load( '../hy_dataset/train_data.npy' )
    train_labels = np.load( '../hy_dataset/train_labels.npy' )
    train_sizes  = np.load( '../hy_dataset/train_sizes.npy' )
    test_data   = np.load( '../hy_dataset/test_data.npy' )
    test_labels = np.load( '../hy_dataset/test_labels.npy' )
    test_sizes  = np.load( '../hy_dataset/test_sizes.npy' )'''

def prepareData():
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


def evaluate(model, test_data, test_labels, test_sizes, logPath):
    accuracy_list = list()
    #confusion_matrix = np.zeros((9,9))
    # for each video part
    for videoId in range(test_data.shape[0]):
        video_data = np.zeros((int(test_sizes[videoId]), stackLen, 64*64))
        video_labels = np.zeros((int(test_sizes[videoId]), 9))
        
        # for each frame of the video
        for frameId in range(int(test_sizes[videoId]) - stackLen):
            if np.sum(test_labels[videoId, frameId]) == 0:
                continue
            stack_labels = list()
            # groups stackLen frames
            for j in range(stackLen):
                video_data[frameId,j] = test_data[videoId, frameId+j]
                stack_labels += [np.argmax(test_labels[videoId, frameId+j])]
            # label referred to the last frame of the stack
            label = stack_labels[stackLen-1]
            video_labels[frameId, label] = 1
        # prepares data
        video_data = video_data.transpose((0,2,1))
        video_data = np.reshape(video_data, (int(test_sizes[videoId]), 64, 64, stackLen))
        # runs networkon test batch to obtain its accuracy and confusion matrix
        loss, acc = model.evaluate(video_data, video_labels,
                       batch_size=1, verbose=0)
        # concatenates accuracy and sums confusion matrix
        accuracy_list += [acc]
    mean_accuracy = np.mean(accuracy_list)
        #confusion_matrix += cm_batch
    # saves test accuracy to file
    with open(logPath, 'a') as logFile:
        logFile.write(str(np.mean(accuracy_list)) + '\n') 
    # presents result
    print("test accuracy %g"%(np.mean(accuracy_list)))
    #print('confusion matrix:\n', confusion_matrix)
    return mean_accuracy

if __name__ == '__main__':
    input_shape=(64,64,stackLen)
    num_classes=9
    
    # Prepare data
    train_data, train_labels, train_sizes, \
            test_data, test_labels, test_sizes = prepareData()
    

    print('train_data_distribute:',np.sum(np.sum(train_labels,axis=0),axis=0), np.sum(train_labels[:][:][:]))
    print('test_data_distribute:',np.sum(np.sum(test_labels,axis=0),axis=0),np.sum(test_labels[:][:][:]))
    print('train_test_percentage:',np.true_divide(np.sum(np.sum(test_labels,axis=0),axis=0),np.sum(np.sum(train_labels,axis=0),axis=0)))
    print ('train_data:',train_data.shape,train_labels.shape,train_sizes[:])
    print ('test_data:',test_data.shape,test_labels.shape,test_sizes[:])


    # Build network model
    #model = tiny_XCEPTION(input_shape,num_classes)
    model = mini_XCEPTION(input_shape,num_classes)

    ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=ada,
                  metrics=['accuracy'])
    model.summary() 
    file_path = '../trained_model/' + 'fraFred_1_mini_XCEPTION.hdf5' 
    logPath = '../hy_results/log_fraFred_1_mini_XCEPTION.txt'

    train_batch_size = 32
    #for epoch in range(12501):
    accuracy_best = 0
    for epoch in range(10000):
        batch_data = np.zeros((train_batch_size, stackLen, 64*64))
        batch_labels = np.zeros((train_batch_size, 9))
        # selects video parts randomly to compose a batch for training
        videosId = np.random.choice(train_data.shape[0], train_batch_size)
        # for each video part selected
        #print('start1')
        for i in range(train_batch_size):
            # Select a frame of the video
            frameId = np.random.randint(0,train_sizes[videosId[i]]-stackLen)
            # The label must be a valid activity
            #while np.sum(train_labels[frameId+stackLen-1]) == 0.0:
            '''while np.sum(train_labels[videosId][frameId+stackLen-1]) == 0.0:
                frameId = np.random.randint(0,train_sizes[videosId[i]]-stackLen)'''
            stack_labels = list()
            # groups stackLen frames
            for j in range(stackLen):
                batch_data[i,j] = train_data[videosId[i], frameId+j]
                stack_labels += [np.argmax(train_labels[videosId[i], frameId+j])]
            # label referred to the last frame of the stack
            label = stack_labels[stackLen-1]
            batch_labels[i, label] = 1
        # prepare batch for training
        batch_data = batch_data.transpose((0,2,1))
        batch_data = np.reshape(batch_data, (train_batch_size, 64, 64, stackLen))
        # train the selected batch
        #print('start2')
        model.train_on_batch(batch_data, batch_labels,
                             class_weight=None, sample_weight=None)
        #print('start3')
        if epoch%1 == 0:
            print('Step:', epoch)
            mean_accuracy=evaluate(model, test_data, test_labels, test_sizes, logPath)
        if mean_accuracy > accuracy_best:
            accuracy_best = mean_accuracy
            model.save(file_path)
        print('best_accuracy:',accuracy_best)
        








