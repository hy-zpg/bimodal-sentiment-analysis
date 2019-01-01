import numpy as np
import argparse
from i3d_inception import Inception_Inflated3d
from mini_xception import mini_XCEPTION
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from keras.layers.core import Reshape
import numpy as np
import cv2 

emotion_model_path = '../emotion_recognition/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66-no-top.h5'
network_frame_path = './network_frame/'

action_input_shape = (79, 224, 224, 3)
emotion_input_shape = (64,64,1)
action_num_classes = 400
emotion_num_classes = 7

'''action_feature_model = Inception_Inflated3d(
                include_top=False,
                weights='rgb_kinetics_only',
                input_shape=action_input_shape,
                classes=action_num_classes)'''
#action_feature_model.summary()
emotion_feature_model = mini_XCEPTION(
						input_shape=emotion_input_shape, 
						num_classes=emotion_num_classes,
						include_top=False,
						weights_path = emotion_model_path,
						l2_regularization=0.01)

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def generate_img_for_emotion(file_path,emotion_target_size):
	img = cv2.imread(file_path)
	cv2.imshow('test',img)
	cv2.waitKey(1000)
	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray_face = cv2.resize(gray_image, (emotion_target_size))
	gray_face = preprocess_input(gray_face, False)
	gray_face = np.expand_dims(gray_face, 0)#(64.64)-->(1,64,64) in order to adapt to the tensorflow requirements
	gray_face = np.expand_dims(gray_face, -1)#(1,64,64) -->(1,64,64,1)
	return gray_face
	
'''
file_path ='test.jpeg'
emotion_target_size = (64,64)
gray_face = generate_img_for_emotion(file_path,emotion_target_size)
out_put = emotion_feature_model.predict(gray_face)
feature = np.reshape(out_put,(2048))
print('feature.shape:',feature.shape)
plot_model(action_feature_model, to_file=network_frame_path + 'action_feature_model.png')                                       
plot_model(emotion_feature_model, to_file=network_frame_path + 'emotion_feature_model.png')
'''

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

detection_model_path = '../emotion_recognition/trained_models/detection_models/haarcascade_frontalface_default.xml'
file_path ='test1.jpg'
emotion_target_size = (64,64)
emotion_offsets = (20, 40)


def generate_face_area(file_path,detection_model_path):
	img = cv2.imread(file_path)
	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.imshow('test_1',gray_image)
	cv2.waitKey(5000)
	print('gray_image.shape',gray_image.shape)
	face_detection = load_detection_model(detection_model_path)
	faces = detect_faces(face_detection, gray_image)
	#print('face_coordination:',len(np.array(faces)))
	print('face_coordination:',len(faces))
	if (len(faces)!=0): 
            for face_coordinates in faces:
                x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                gray_face = gray_image[y1:y2, x1:x2]
                try:
                    gray_face = cv2.resize(gray_face, (emotion_target_size))
                except:
                    continue
                cv2.imshow('test',gray_face)
                cv2.waitKey(5000)
                gray_face = cv2.resize(gray_face, (emotion_target_size))
                gray_face = preprocess_input(gray_face, False)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                return gray_face
                
gray_face = generate_face_area(file_path,detection_model_path)
#cv2.imshow('test',gray_face)
#cv2.waitKey(1000)
print('gray_face.shape:',len(gray_face))

        	

#gray_face = preprocess_input(gray_face, False)
#gray_face = np.expand_dims(gray_face, 0)
#gray_face = np.expand_dims(gray_face, -1)
#print('gray_face.shape:',gray_face.shape)



'''gray_face = gray_image[faces[0][0]:faces[0][0]+faces[0][2],faces[0][1]:faces[0][1]+faces[0][3]]
cv2.imshow('face',gray_face)
cv2.waitKey(10000)
print('faces.shape:',gray_face.shape)'''


'''while(np.array(faces).shape[0]!=0): 
	#x1, x2, y1, y2 = apply_offsets(faces, emotion_offsets)
	#gray_face = gray_image[y1:y2, x1:x2]
	#print(gray_face.shape)
    #cv2.imshow('face',gray_face)
    #cv2.waitKey(1000)'''

#print('gray_face.shape',gray_face.shape)
#cv2.imshow('face',gray_face)
#cv2.waitKey(1000)
'''try:
    gray_face = cv2.resize(gray_face, (emotion_target_size))
except:
    continue'''
'''gray_face = preprocess_input(gray_face,False)
gray_face = np.expand_dims(gray_face,0)
gray_face = np.expand_dims(gray_face,-1)
print(gray_face.shape)
break'''


