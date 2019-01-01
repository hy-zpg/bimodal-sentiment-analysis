import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
  

stackLen = 3
maxPartsTrain = 77#obtained from clean_train_size 84
#second recycle--videoId,frameId,partId,frameId_clean: 41 24 66 0
maxPartsTest = 20#obtained from clean_test_size13
#second recycle--videoId,frameId,partId,frameId_clean: 19 35 30 19
maxFrames = 30
#first recycle--videoId,frameId,partId,frameId_clean: 10 33 17 22  

#train_50:77 20 [ 0.21052632  0.08571429  0.15384615  0.35833333  0.17391304  0.33802817 0.33333333  0.37254902  0.41176471]
#train_55:84 13 [ 0.16949153  0.2         0.11111111  0.17266187  0.10958904  0.15853659 0.06024096  0.04477612  0.152     ]
#train_52:83 14 [ 0.104       0.04587156  0.23287671  0.15602837  0.17391304  0.30136986 0.23943662  0.22807018  0.22033898]
#train_55_1:86 11 [ 0.12195122  0.11764706  0.11111111  0.14788732  0.10958904  0.14457831 0.12820513  0.12903226  0.13385827]
#train_50_1:79 18 [ 0.23214286  0.23913043  0.23287671  0.23484848  0.265625    0.23376623 0.25714286  0.27272727  0.24137931]
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


def prepareData():
    '''train_data   = np.load( '/home/yanhong/Downloads/next_step/JPL_First-Person_Interaction activity_video/data/original/train_data.npy' )
    train_labels = np.load( '/home/yanhong/Downloads/next_step/JPL_First-Person_Interaction activity_video/data/original/train_labels.npy' )
    train_sizes  = np.load( '/home/yanhong/Downloads/next_step/JPL_First-Person_Interaction activity_video/data/original/train_sizes.npy' )
    test_data   = np.load( '/home/yanhong/Downloads/next_step/JPL_First-Person_Interaction activity_video/data/original/validation_data.npy' )
    test_labels = np.load( '/home/yanhong/Downloads/next_step/JPL_First-Person_Interaction activity_video/data/original/validation_labels.npy' )
    test_sizes  = np.load( '/home/yanhong/Downloads/next_step/JPL_First-Person_Interaction activity_video/data/original/validation_sizes.npy' )'''
    train_data   = np.load( '../data/train_data_50.npy' )
    train_labels = np.load( '../data/train_labels_50.npy' )
    train_sizes  = np.load( '../data/train_sizes_50.npy' )
    test_data   = np.load( '../data/validation_data_12.npy' )
    test_labels = np.load( '../data/validation_labels_12.npy' )
    test_sizes  = np.load( '../data/validation_sizes_12.npy' )

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



if __name__ == '__main__':
    input_shape=(64,64,stackLen)
    num_classes=9
    
    # Prepare data
    train_data, train_labels, train_sizes, \
            test_data, test_labels, test_sizes = prepareData()
    print ('train_data:',train_data.shape,train_labels.shape,train_sizes[:])
    print ('test_data:',test_data.shape,test_labels.shape,test_sizes[:])
    '''fig = plt.figure()
    ax = Axes3D(fig)
    train_x = np.arange(0, train_data.shape[0]-1, 1)
    train_y = np.arange(0, train_data.shape[1]-1, 1)
    train_x,train_y = np.meshgrid(train_x,train_y)
    train_z = train_data[train_x,train_y,:]
    ax.plot_surface(train_x,train_y,train_z,rstride=1, cstride=1, cmap='rainbow')
    plt.show()'''

    fig = plt.figure()
    train_x = np.arange(0, train_data.shape[0]-1, 1)
    train_y = np.arange(0, train_data.shape[1]-1, 1)
    train_z = train_data[train_x[0:5],train_y[0:5],0:512]
    print(train_data[0][0][:].shape)
    print(np.max(train_data[0][0][:]))
    print(np.min(train_data[0][0][:]))
    pic_train = train_d.reshape(77*30,4096)
    plt.hist(pic_train[:])
    plt.show()

    '''ax=plt.subplot(111,projection='3d') 
    ax.scatter(train_x[0:5],train_y[0:5],train_z[:],c='y') 
    ax.set_zlabel('activity')
    ax.set_ylabel('frame')
    ax.set_xlabel('pixel')
    plt.show()'''


    '''print('train_data_distribute:',np.sum(np.sum(train_labels,axis=0),axis=0), np.sum(train_labels[:][:][:]))
    print('test_data_distribute:',np.sum(np.sum(test_labels,axis=0),axis=0),np.sum(test_labels[:][:][:]))
    print('train_test_percentage:',np.true_divide(np.sum(np.sum(test_labels,axis=0),axis=0),np.sum(np.sum(train_labels,axis=0),axis=0)))
    print ('train_data:',train_data.shape,train_labels.shape,train_sizes[:])
    print ('test_data:',test_data.shape,test_labels.shape,test_sizes[:])'''
