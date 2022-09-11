from tkinter import N
import mediapipe as mp
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import time
import sklearn

from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

# Constants
t = 2
t1 = 3
r = 2

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Path for exported data
DATA_PATH = os.path.join('MP_Data')

# Actions wanted to be detected
actions = np.array(['right', 'wrong', 'learn'])

# 30 videos worth of data
no_sequences = 30

# 30 frames in length per video
sequence_length = 30

# 1 folder for each sequence
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

def mediapipeDetection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False # Image is no longer writeable

    results = model.process(image) # Makes prediction
    image.flags.writeable = True # Image is noe writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Color conversion

    return image, results


def drawLandmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def drawStyledLandmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1), 
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=t1, circle_radius=r), 
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=t1, circle_radius=r))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=t, circle_radius=r), 
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=t, circle_radius=r))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 76), thickness=t, circle_radius=r), 
        mp_drawing.DrawingSpec(color=(245, 66, 250), thickness=t, circle_radius=r))

def extractKeypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, face, lh, rh])

labelMap = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(labelMap[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

res = [.7, .2, .1]
actions[np.argmax(res)]

# Loss function used because multi class classification model
model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

# cap = cv2.VideoCapture(0)

# with mp_holistic.Holistic(min_detection_confidence = 0.8, min_tracking_confidence = 0.8) as holistic:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         frame = cv2.flip(frame, 1)

#         # Make detections
#         image, results = mediapipeDetection(frame, holistic)

#         drawStyledLandmarks(image, results)

    # # Loop through actions
    # for action in actions:
    #     # Loop through sequences (videos)
    #     for sequence in range(no_sequences):
    #         # Loop through video length (sequence length)
    #         for frame_num in range(sequence_length):

                # ret, frame = cap.read()
                # frame = cv2.flip(frame, 1)

                # # Make detections
                # image, results = mediapipeDetection(frame, holistic)

                # drawStyledLandmarks(image, results)

                # Apply wait logic
                # if frame_num == 0:
                #     cv2.putText(image, 'STARTING COLLECTION', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                #     cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                #     cv2.waitKey(2000) # Waits 2 seconds
                # elif 0xFF == ord('q'):
                #     break
                # else:
                #     cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # # Exports keypoints
                # keypoints = extractKeypoints(results)
                # npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                # np.save(npy_path, keypoints)
                
                # Shows to screen
                # cv2.imshow('Feed', image)
        
#         cv2.imshow('Feed', image)
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()