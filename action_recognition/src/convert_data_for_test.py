import numpy as np
import cv2
import os
import csv


to_gray = True
r, c = 64, 64
frameFreq = 10 
videoFrames = 30
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
    # Activity labels: 4 + 3 * activity count
    # Activity beginins: 5 + 3 * activity count
    # Activity endings: 6 + 3 * activity count
    label = np.zeros((nClasses))
    # default: frame does not belong to any class
    label[nClasses-1] = 1
    # the csv file stores, in these cells, the times in which frames end
    activityEndFrame = labelsList[videoNumber][6+3*activityCount]
    # if there are activities to read
    if activityEndFrame != '':
        if frameNumber > int(activityEndFrame):
            # check if the activity has already ended
            activityCount += 1

    # the csv file stores, in these cells, the times in which frames begin
    activityBeginFrame = labelsList[videoNumber][5+3*activityCount]
    # if there are activities to read
    if activityBeginFrame != '':
        # register its label in the label array
        if frameNumber >= int(activityBeginFrame):
            index = int(labelsList[videoNumber][4+3*activityCount])
            #finding the class index:4-point  ###from the act_type to find the label
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
    # reads all frames of the video
    while ret:
        # frameNumber is related to each frame on the original video
        frameNumber += 1
        ret, frame = videoCapture.read()
        # sample 1 video in each frameFreq sequence
        if not ret or frameNumber % frameFreq != 0:
            continue
        # converts video to grayscale
        if( to_gray ):
            frame = cv2.cvtColor( frame , cv2.COLOR_BGR2GRAY )
            # normalizes video
            cv2.normalize(frame,frame,0,255,cv2.NORM_MINMAX)
            #cv2.namedWindow("current_img") 
            #cv2.imshow('current_img:',frame)
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

        


dataDirectories += ['/home/yanhong/Downloads/next_step/JPL_First-Person_Interaction activity_video/dataset/robot_interaction_test']
labelsPath = '/home/yanhong/Downloads/next_step/JPL_First-Person_Interaction activity_video/data/robot_interaction_labels.csv'
# obtain arrays with all videos, labels and video sizes
videosArray, labelsArray, videoSizes = getAllVideos(dataDirectories, labelsPath)

# Split array, randomly, into train and test sets
videosIds = np.array(range(numVideos))
np.random.shuffle(videosIds)

np.save('../data/test_data.npy', videosArray[videosIds[trainSize:]])
np.save('../data/test_labels.npy', labelsArray[videosIds[trainSize:]])
np.save('../data/test_sizes.npy', videoSizes[videosIds[trainSize:]])
