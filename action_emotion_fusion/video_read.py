import cv2
import numpy as np



detection_model_path = '../emotion_recognition/trained_models/detection_models/haarcascade_frontalface_default.xml'

emotion_offsets = (20, 40)
NUM_FRAMES=40
FRAME_WIDTH=224
FRAME_HEIGHT=224

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model


def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


face_detection = load_detection_model(detection_model_path)

def emotion_frame_process(original_frame,emotion_target_size=(64,64)):
    gray_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    print('original.shape:',gray_frame.shape)
    faces = detect_faces(face_detection, gray_frame)
    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_frame[y1:y2, x1:x2]
        print('face_area.shape:',gray_face.shape)
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        gray_face = preprocess_input(gray_face, False)
        gray_face = np.expand_dims(gray_face, -1)#(1,64,64) -->(1,64,64,1)
        print('gray_face.shape:',gray_face.shape)
        return gray_face
    




print('hello_2')
video_capture = cv2.VideoCapture('./test_data/Calins_gratuits_a_Paris_-_Free_Hugs_France_-_version_longue_hug_u_cm_np2_ba_med_21.avi')
action_frames = [] 
emotion_frames = []
print('hello_3')
num_frame=0
print(video_capture.isOpened())
while (video_capture.isOpened()):
    ret, frame = video_capture.read()
    if ret is False:
        continue
    process_emotion_frame = emotion_frame_process(frame)
    cv2.normalize(frame,frame,0,255,cv2.NORM_MINMAX)
    process_action_frame = cv2.resize(frame, (FRAME_HEIGHT, FRAME_WIDTH))
    #print('process_emotion_frame:',process_emotion_frame.shape)
    print('process_action_frame:',process_action_frame.shape)
    emotion_frames.append(process_emotion_frame)
    action_frames.append(process_action_frame)
    num_frame+=1
    if num_frame>=NUM_FRAMES:
        action_frames = np.array(action_frames)
        emotion_frames = np.array(emotion_frames)
        break
print('action.shape:',action_frames.shape)
print('emotion.shape:',emotion_frames.shape)
np.save('./test_data/action_data_1.npy', action_frames)
np.save('./test_data/emotion_data_1.npy', emotion_frames)
