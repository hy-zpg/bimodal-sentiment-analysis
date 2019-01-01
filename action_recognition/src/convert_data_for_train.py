# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 19:42:43 2017

"""
#generates the .npy files in directory data/, based on the original videos, 
#which must be located on the directory "dataset/"
import numpy as np
import cv2
import os
import csv


to_gray = True
r, c = 64, 64
frameFreq = 1
#frameFreq = 10 #30fps --> 3fps
videoFrames = 700
#videoFrames = 70
numVideos = 62
trainSize = 54
nClasses = 10

# read csv file with the labels
#result:labellist[0][0:18]: ['Aggarwal', '1', '632', '4', '3', '168', '226', '2', '239', '287', '0', '316', '440', '5', '441', '523', '', '', '', '', '', '']
'''labellist[i][:]: ['Aggarwal', '1', '632', '4', '3', '168', '226', '2', '239', '287', '0', '316', '440', '5', '441', '523', '', '', '', '', '', ''] 
labellist[i][:]: ['Aggarwal', '2', '444', '2', '3', '158', '209', '5', '305', '371', '', '', '', '', '', '', '', '', '', '', '', ''] 

labellist[i][:]: ['Aggarwal', '3', '260', '1', '1', '144', '212', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] 

labellist[i][:]: ['Aggarwal', '4', '389', '4', '3', '67', '104', '0', '186', '257', '1', '258', '330', '', '', '', '', '', '', '', '', ''] 

labellist[i][:]: ['Aggarwal', '5', '392', '4', '3', '50', '126', '6', '163', '208', '7', '209', '264', '5', '301', '361', '', '', '', '', '', ''] 

labellist[i][:]: ['Aggarwal', '6', '293', '3', '3', '64', '114', '4', '114', '143', '8', '143', '220', '', '', '', '', '', '', '', '', ''] 

labellist[i][:]: ['Aggarwal', '7', '378', '4', '3', '37', '79', '6', '99', '146', '7', '147', '182', '8', '183', '298', '', '', '', '', '', ''] 

labellist[i][:]: ['Birgi', '1', '552', '5', '2', '89', '149', '3', '168', '217', '2', '218', '256', '0', '315', '418', '1', '419', '502', '', '', ''] 

labellist[i][:]: ['Birgi', '2', '235', '2', '3', '53', '91', '5', '123', '157', '', '', '', '', '', '', '', '', '', '', '', ''] 

labellist[i][:]: ['Birgi', '3', '259', '1', '1', '118', '210', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''] 

labellist[i][:]: ['Birgi', '4', '324', '3', '3', '36', '62', '2', '66', '112', '0', '168', '242', '', '', '', '', '', '', '', '', ''] 

labellist[i][:]: ['Birgi', '5', '305', '3', '4', '83', '123', '6', '124', '172', '7', '173', '195', '', '', '', '', '', '', '', '', ''] '''
def readLabels(path):
    labelsList = list()
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            row += [''] + [''] + [''] 
            labelsList += [row]
    #for i in range(63):
        #print ('labellist[i][:]:',labelsList[i][:],'\n')
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
            #cv2.normalize(frame,frame,0,1,cv2.NORM_MINMAX)
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
            print('if open:',videoCapture.isOpened())
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
dataDirectories = ['/home/yanhong/Downloads/next_step/JPL_First-Person_Interaction activity_video/dataset/robot_interaction_part1']
dataDirectories += ['/home/yanhong/Downloads/next_step/JPL_First-Person_Interaction activity_video/dataset/robot_interaction_part2']
labelsPath = '/home/yanhong/Downloads/next_step/JPL_First-Person_Interaction activity_video/data/robot_interaction_labels.csv'
# obtain arrays with all videos, labels and video sizes
videosArray, labelsArray, videoSizes = getAllVideos(dataDirectories, labelsPath)

# Split array, randomly, into train and test sets
'''trainSize = 42
videosIds = np.array(range(numVideos))
np.random.shuffle(videosIds)
a=np.sum(np.sum(labelsArray[videosIds[0:trainSize]],axis=0),axis=0)
b=np.sum(np.sum(labelsArray[videosIds[trainSize:]],axis=0),axis=0)
print(np.logical_or(a[:9].any()==0,b[:9].any()==0))
percentage = np.true_divide(b[:9],a[:9])
percentage_below_compare = percentage - 0.2*np.ones(9)
percentage_above_compare = 0.25*np.ones(9) - percentage
standard_vector = np.zeros(9)
comparison = (percentage_below_compare>=standard_vector)&(percentage_above_compare>=standard_vector)
np.save('../data/fraFred_1/train_data.npy', videosArray[videosIds[0:trainSize]])
np.save('../data/fraFred_1/train_labels.npy', labelsArray[videosIds[0:trainSize]])
np.save('../data/fraFred_1/train_sizes.npy', videoSizes[videosIds[0:trainSize]])
np.save('../data/fraFred_1/validation_data.npy', videosArray[videosIds[trainSize:]])
np.save('../data/fraFred_1/validation_labels.npy', labelsArray[videosIds[trainSize:]])
np.save('../data/fraFred_1/validation_sizes.npy', videoSizes[videosIds[trainSize:]])
print('final_train_data_distribute:',np.sum(np.sum(labelsArray[videosIds[0:trainSize]],axis=0),axis=0), np.sum(labelsArray[videosIds[0:trainSize]][:][:]))
print('final_test_data_distribute:',np.sum(np.sum(labelsArray[videosIds[trainSize:]],axis=0),axis=0),np.sum(labelsArray[videosIds[trainSize:]][:][:]))
print('final_train_test_percentage:',np.true_divide(b,a))'''


while (1):
    trainSize = np.random.randint(45,50)
    videosIds = np.array(range(numVideos))
    np.random.shuffle(videosIds)
    a=np.sum(np.sum(labelsArray[videosIds[0:trainSize]],axis=0),axis=0)
    b=np.sum(np.sum(labelsArray[videosIds[trainSize:]],axis=0),axis=0)
    print(np.logical_or(a[:9].any()==0,b[:9].any()==0))
    if (np.logical_or(a[:9].any()==0,b[:9].any()==0)):
        continue
    percentage = np.true_divide(b[:9],a[:9])
    percentage_below_compare = percentage - 0.3*np.ones(9)
    percentage_above_compare = 0.4*np.ones(9) - percentage
    standard_vector = np.zeros(9)
    comparison = (percentage_below_compare>=standard_vector)&(percentage_above_compare>=standard_vector)
    print('train_data_distribute:',np.sum(np.sum(labelsArray[videosIds[0:trainSize]],axis=0),axis=0), np.sum(labelsArray[videosIds[0:trainSize]][:][:]))
    print('test_data_distribute:',np.sum(np.sum(labelsArray[videosIds[trainSize:]],axis=0),axis=0),np.sum(labelsArray[videosIds[trainSize:]][:][:]))
    print('train_test_percentage:',np.true_divide(b,a))

    if comparison.all()==True:
        np.save('../data/fraFred_1/train_data.npy', videosArray[videosIds[0:trainSize]])
        np.save('../data/fraFred_1/train_labels.npy', labelsArray[videosIds[0:trainSize]])
        np.save('../data/fraFred_1/train_sizes.npy', videoSizes[videosIds[0:trainSize]])
        np.save('../data/fraFred_1/validation_data.npy', videosArray[videosIds[trainSize:]])
        np.save('../data/fraFred_1/validation_labels.npy', labelsArray[videosIds[trainSize:]])
        np.save('../data/fraFred_1/validation_sizes.npy', videoSizes[videosIds[trainSize:]])
        print('final_train_data_distribute:',np.sum(np.sum(labelsArray[videosIds[0:trainSize]],axis=0),axis=0), np.sum(labelsArray[videosIds[0:trainSize]][:][:]))
        print('final_test_data_distribute:',np.sum(np.sum(labelsArray[videosIds[trainSize:]],axis=0),axis=0),np.sum(labelsArray[videosIds[trainSize:]][:][:]))
        print('final_train_test_percentage:',np.true_divide(b,a))
        break
