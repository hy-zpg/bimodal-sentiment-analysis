import keras
import cv2
import os
import numpy as np
import pickle 
import scipy.misc
import tensorflow as tf
import matplotlib.pyplot as plt 
from statistics import mode
from keras.models import Model, load_model, Sequential
from keras import layers
from keras.layers import Activation, Convolution2D, Dropout, Conv2D, Dense
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras.regularizers import l2
from keras.optimizers import Adadelta
from keras.utils import np_utils
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from scipy import ndimage 
from keras.utils.vis_utils import plot_model 

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

detection_model_path = './emotion_recognition/trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = './emotion_recognition/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = './emotion_recognition/trained_models/gender_models/simple_CNN.81-0.96.hdf5'
action_model_path = './action_recognition/trained_model/model_generate_55_stable_0.84.hdf5'
logPath = './realtime_results/test_result.txt'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
action_labels = ['hand_shake', 'hug', 'wave', 'stand up', 'point', 'punch', 'reach', 'throw', 'run']
font = cv2.FONT_HERSHEY_SIMPLEX

frame_window = 10
gender_offsets = (30, 60)
emotion_offsets = (20, 40)
action_offsets = (0,40)

face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)
action_classifier = load_model(action_model_path,compile=False)

emotion_target_size = emotion_classifier.input_shape[1:3] #(64,64)
gender_target_size = gender_classifier.input_shape[1:3] #(48,48)
action_target_size = action_classifier.input_shape[1:3] #(64,64)
stackLen = 3 #3

gender_window = []
emotion_window = []

cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
ret = video_capture.isOpened()
print('if open:',(video_capture.isOpened()))
frameId = -1
newframeId = -1
while ret:
    bgr_image = video_capture.read()[1]

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray_image_action = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    cv2.normalize(gray_image_action,gray_image_action,0,255,cv2.NORM_MINMAX)
    gray_action_image = cv2.resize(gray_image_action, (action_target_size))
    faces = detect_faces(face_detection, gray_image)

    # recognize the emotion and gender
    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
        rgb_face = rgb_image[y1:y2, x1:x2]

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            rgb_face = cv2.resize(rgb_face, (gender_target_size))
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        gray_face = preprocess_input(gray_face, False)
        gray_face = np.expand_dims(gray_face, 0)#(64.64)-->(1,64,64) in order to adapt to the tensorflow requirements
        gray_face = np.expand_dims(gray_face, -1)#(1,64,64) -->(1,64,64,1)
        #print('gray_face.shape:',gray_face.shape)#(1,64,64,1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))#return the probility vector
        #print('emotion_label_arg:',emotion_label_arg)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)
        rgb_face = np.expand_dims(rgb_face, 0)
        #print('rgb_face.shape:',rgb_face.shape)#(1,48,48,3)
        rgb_face = preprocess_input(rgb_face, False)
        gender_prediction = gender_classifier.predict(rgb_face)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]
        gender_window.append(gender_text)
        if len(gender_window) > frame_window:
            emotion_window.pop(0)
            gender_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
            gender_mode = mode(gender_window)
        except:
            continue
        if gender_text == gender_labels[0]:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, gender_mode,
                  color, 0, -20, 1, 1)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)
	#with open(logPath, 'a') as logFile:
		#logFile.write('emotion:' + str(emotion_text) + 'gender:' + str(gender_text) +'\n')

    # recognize the action
    test_framesArray = np.zeros((stackLen,64,64))
    frameId += 1
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
        cv2.putText(rgb_image,action_text,action_offsets,font,1.2,(255,255,255),2) 
        with open(logPath, 'a') as logFile:
                logFile.write(str(action_text) + '\n')  



   

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break





