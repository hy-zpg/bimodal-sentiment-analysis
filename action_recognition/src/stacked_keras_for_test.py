#when the video has vonverted into frames, cleanSet to delete no-label data and the evaluate function to combine continuous 3 frame
import numpy as np
import keras
import cv2
from keras import layers
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Activation, Convolution2D, Dropout, Conv2D, Dense
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras.regularizers import l2
from keras.optimizers import Adadelta
from keras.utils import np_utils
import pickle 
import scipy.misc
from scipy import ndimage
import matplotlib.pyplot as plt  
from keras.utils.vis_utils import plot_model   
import os
import csv
from statistics import mode
from stacked_keras_model import model_generate


def real_time_evaluate(action_classifier, videoCapture, stackLen, logPath, action_target_size):
    frameId = -1
    newframeId = -1
    ret = videoCapture.isOpened()
    print('if open:',(videoCapture.isOpened()))
    #while True:
    while ret:
        test_framesArray = np.zeros((stackLen,64,64))
        frameId += 1
        bgr_image = videoCapture.read()[1]
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        cv2.normalize(gray_image,gray_image,0,255,cv2.NORM_MINMAX)
        gray_action_image = cv2.resize(gray_image, (action_target_size))
        if (frameId+1) % 10 == 0:
            newframeId += 1
            test_framesArray[newframeId,:] = gray_action_image
        if frameId == 29:
            frameId = -1
            newframeId = -1
            print('one test batch has converted!')
            test_framesArray = test_framesArray.transpose((0,2,1))
            test_framesArray = np.expand_dims(test_framesArray, 0)
            test_framesArray = np.reshape(test_framesArray, (1, 64, 64, stackLen))
            action_label_arg = np.argmax(action_classifier.predict(test_framesArray))#return the probility vector
            action_text = action_labels[action_label_arg]
            with open(logPath, 'a') as logFile:
                logFile.write(str(action_text) + '\n') 
            font=cv2.FONT_HERSHEY_SIMPLEX#使用默认字体
            font=cv2.FONT_HERSHEY_SIMPLEX#使用默认字体
            cv2.putText(rgb_image,action_text,(0,40),font,1.2,(255,255,255),2)#添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
            #cv.PutText(rgb_image, action_text, (30,30), font, (0,255,0))       
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame', bgr_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



if __name__ == '__main__':
    stackLen = 3
    action_model_path = '../trained_model/model_generate_55_stable_0.84.hdf5'
    logPath = '../hy_results/test_result.txt'
    action_classifier = load_model(action_model_path,compile=False)
    action_target_size = action_classifier.input_shape[1:3]
    #print('action_target_size.shape',action_target_size.shape)
    action_labels = ['hand_shake', 'hug', 'wave', 'stand up', 'point', 'punch', 'reach', 'throw', 'run']
    cv2.namedWindow('window_frame')
    videoCapture = cv2.VideoCapture(0)
    #videoCapture = cv2.VideoCapture('/home/yanhong/Downloads/next_step/JPL_First-Person_Interaction activity_video/dataset/robot_interaction_part2')
    real_time_evaluate(action_classifier, videoCapture, stackLen, logPath, action_target_size)
