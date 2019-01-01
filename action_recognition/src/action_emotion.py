import numpy as np
import cv2
import os
import csv


to_gray = True
r, c = 64, 64
frameFreq = 10 #30fps --> 3fps
videoFrames = 70
numVideos = 62
trainSize = 42
nClasses = 10

def readLabels(path):
    labelsList = list()
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            row += [''] + [''] + [''] 
            labelsList += [row]
    return labelsList
    
    
def getFrameLabel(labelsList, activityCount, videoNumber, frameNumber):
    label = np.zeros((nClasses))
    label[nClasses-1] = 1
    activityEndFrame = labelsList[videoNumber][6+3*activityCount]
    if activityEndFrame != '':
        if frameNumber > int(activityEndFrame):
            activityCount += 1
    activityBeginFrame = labelsList[videoNumber][5+3*activityCount]
    if activityBeginFrame != '':
        if frameNumber >= int(activityBeginFrame):
            index = int(labelsList[videoNumber][4+3*activityCount])
            label[index] = 1
            label[nClasses-1] = 0
    return label, activityCount


def convertVideo(videoCapture, labelsList, videoNumber, videoName='unknown'):
    print('Converting', videoName)
    framesArray = np.zeros((videoFrames, r*c))
    videoLabelsArray = np.zeros((videoFrames, nClasses))
    ret = True
    frameNumber = -1
    newFrameNumber = 0
    activityCount = 0
    while ret:
        frameNumber += 1
        ret, frame = videoCapture.read()
        if not ret or frameNumber % frameFreq != 0:
            continue
        if( to_gray ): 
            frame = cv2.cvtColor( frame , cv2.COLOR_BGR2GRAY )
            cv.imshow('current frame',frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            cv2.normalize(frame,frame,0,255,cv2.NORM_MINMAX)
        # resizes to array
        frameArray = cv2.resize(frame, (r, c))
        # puts frame in the matrix
        framesArray[newFrameNumber] = np.reshape(frameArray, (r * c))
        # label is obtained in another function
        label, activityCount = getFrameLabel(labelsList, activityCount,
                                             videoNumber, frameNumber)
        videoLabelsArray[newFrameNumber] = label
        # newFrameNumber is related only to the frames that were sampled
        newFrameNumber += 1
    print(videoName, 'converted!')
    print('framesarray',framesArray.shape)#70*4096
    print('videolabelsarray',videoLabelsArray.shape)#70*10
    return framesArray, videoLabelsArray


# reads all videos and organizes them
def getAllVideos(dataDirectories, labelsPath):
    # initializes empty arrays
    videosArray = np.zeros((numVideos, videoFrames, r*c))
    labelsList = readLabels(labelsPath)
    labelsArray = np.zeros((numVideos, videoFrames, nClasses))
    videoSizes = np.zeros((numVideos))
    videoNumber = 0
    # the videos are organized into 2 directories. Explore both.
    for directory in dataDirectories:
        ####obtaining the file names
        fileNames = os.listdir(directory)
        fileNames.sort()
        # each filename is related to a video
        for fileName in fileNames:
            # read video
            videoCapture = cv2.VideoCapture(directory+'/'+fileName)
            # obtains frames and labels
            videosArray[videoNumber], labelsArray[videoNumber] = convertVideo(
                videoCapture, labelsList, videoNumber, fileName)
            # finished reading video
            videoCapture.release()
            # as the videos are stored in an array, their lenghts must be remembered
            videoSizes[videoNumber] = int(int(labelsList[videoNumber][2])/frameFreq)
            videoNumber += 1
    print('videosarray',videosArray.shape)#62*70*4096
    print('labelsarray',labelsArray.shape)#62*70*10
    print('videosize',videoSizes)#62
    return videosArray, labelsArray, videoSizes

        
# paths in which the original data is stored
dataDirectories = ['../dataset/robot_interaction_part1']
dataDirectories += ['../dataset/robot_interaction_part2']
labelsPath = '../data/robot_interaction_labels.csv'
# obtain arrays with all videos, labels and video sizes
videosArray, labelsArray, videoSizes = getAllVideos(dataDirectories, labelsPath)
# Split array, randomly, into train and test sets
videosIds = np.array(range(numVideos))
np.random.shuffle(videosIds)
np.save('../dataset/emotion_action_data/train_data.npy', videosArray[videosIds[0:trainSize]])
np.save('../dataset/emotion_action_data/train_labels.npy', labelsArray[videosIds[0:trainSize]])
np.save('../dataset/emotion_action_data/train_sizes.npy', videoSizes[videosIds[0:trainSize]])
np.save('../dataset/emotion_action_data/test_data.npy', videosArray[videosIds[trainSize:]])
np.save('../dataset/emotion_action_data/test_labels.npy', labelsArray[videosIds[trainSize:]])
np.save('../dataset/emotion_action_data/test_sizes.npy', videoSizes[videosIds[trainSize:]])


