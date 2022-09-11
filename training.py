from tkinter import N
import mediapipe as mp
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import time
import sklearn
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

from functions import drawStyledLandmarks, extractKeypoints, mediapipeDetection

# Path for exported data
DATA_PATH = os.path.join('MP_Data')

# Actions wanted to be detected
actions = np.array(['right', 'wrong', 'learn'])

# 30 videos worth of data
no_sequences = 30

# 30 frames in length per video
sequence_length = 30

labelMap = {label:num for num, label in enumerate(actions)}

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(labelMap[action])

print(sequences)
print(labels)

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# 3
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# Softmax gives an array of probabilities that add up to 1
model.add(Dense(actions.shape[0], activation='softmax'))

# model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# model.fit(X_train, y_train, epochs=2000, callbacks = [tb_callback])
model.load_weights('action.h5')
model.summary()

sequence, sentence = [], []
threshold = 0.4

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipeDetection(frame, holistic)
        print(results)

        drawStyledLandmarks(image, results)

        keypoints = extractKeypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]

            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    print(np.argmax(res))
                    sentence.append(actions[np.argmax(res)])
            
            if len(sentence) > 5:
                sentence = sentence[-5:]
        
        cv2.rectangle(image, (0, 0), (1280, 40), (0,0,0), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
