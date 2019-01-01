import keras
import cv2
import os
import numpy as np
import pickle 
import scipy.misc
import tensorflow as tf
from scipy import ndimage
from statistics import mode
import matplotlib.pyplot as plt 
from statistics import mode
from keras.models import load_model
from keras.utils import np_utils
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input 
from keras.utils.vis_utils import plot_model
from i3d_inception import Inception_Inflated3d


print('hello_1')
'''config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)'''

detection_model_path = '../emotion_recognition/trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../emotion_recognition/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = '../emotion_recognition/trained_models/gender_models/simple_CNN.81-0.96.hdf5'
ACTION_LABEL_MAP_PATH = '../keras-kinetics-i3d-action-recognition/data/label_map.txt'
#action_model_path = './action_recognition/trained_model/model_generate_55_stable_0.84.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
action_labels = [x.strip() for x in open(ACTION_LABEL_MAP_PATH, 'r')]
font = cv2.FONT_HERSHEY_SIMPLEX


NUM_FRAMES=79
FRAME_WIDTH=224
FRAME_HEIGHT=224
NUM_RGB_CHANNELS=3
NUM_CLASSES=400

#emotion_frames= np.load('./test_data/emotion_data_1.npy')
action_frames = np.load('./test_data/action_data_1.npy')

face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)
action_classifier = Inception_Inflated3d(
                include_top=True,
                weights='rgb_imagenet_and_kinetics',
                #input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                classes=NUM_CLASSES) 

emotion_target_size = emotion_classifier.input_shape[1:3] #(64,64)
gender_target_size = gender_classifier.input_shape[1:3] #(48,48)
action_target_size = (FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS) #(224,224,3)


#emotion_label_arg = np.argmax(emotion_classifier.predict(emotion_frames))
#final_emotion = mode(emotion_label_arg)
#emotion_text = emotion_labels[final_emotion]
action_frames = np.expand_dims(action_frames,0)
logic_action = action_classifier.predict(action_frames)


# produce softmax output from model logit for class probabilities
logic_action = logic_action[0] # we are dealing with just one example
sample_predictions = np.exp(logic_action) / np.sum(np.exp(logic_action))
sorted_indices = np.argsort(sample_predictions)[::-1]
action_label_arg = sorted_indices[0]
action_text = action_labels[action_label_arg]

#print('emotion:',emotion_text)
print('action:',action_text)













