stackLen = 3
maxPartsTrain = 66
maxPartsTest = 31
maxFrames = 30


def cleanSet(data, labels, sizes, maxParts, maxFrames, shape, nClasses=9):
    data_clean = np.zeros((maxParts, maxFrames, shape))
    labels_clean = np.zeros((maxParts, maxFrames, nClasses))
    sizes_clean = np.zeros((maxParts))
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
    train_data   = np.load( '../data/original/train_data.npy' )
    train_labels = np.load( '../data/original/train_labels.npy' )
    train_sizes  = np.load( '../data/original/train_sizes.npy' )
    test_data   = np.load( '../data/original/test_data.npy' )
    test_labels = np.load( '../data/original/test_labels.npy' )
    test_sizes  = np.load( '../data/original/test_sizes.npy' )

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