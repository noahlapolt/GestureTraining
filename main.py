import os
import csv
import cv2
import keras
import argparse
import math as m
import numpy as np
import mediapipe as mp
from scipy import stats
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D

def _parse_args():
    """
    Command-line arguments.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='create.py')

    # Element change options.
    parser.add_argument('--elements', nargs='+', help='symbole to train for')
    parser.add_argument('--device', default=0, type=int, help='which camera to use')
    parser.add_argument('--train', dest='train', default=False, action='store_true', help='if the model should train or not')
    parser.add_argument('--retrain', dest='retrain', default=False, action='store_true', help='retrains for a specific value')
    parser.add_argument('--sample_size', default=700, type=int, help='the number of images to take for each element to train')
    parser.add_argument('--train_no', dest='train_no', default=False, action='store_true', help='retrains using the data in the folders')
    parser.add_argument('--data_start', default=0, type=int, help='trains new data names files starting at this number')
    parser.add_argument('--user', required=True, type=str, help='the user the data is being trained for')
    
    # Makes output
    args = parser.parse_args()
    return args


class HandTracking:
    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(tuple(locals().values())[1:])
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def find_hands(self, img):
        self.results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                connections = self.mp_hands.HAND_CONNECTIONS
                self.mp_draw.draw_landmarks(img, hand_lms, connections)
        return img

    def get_points(self, lms, points=(0, 5, 4, 8, 20)):
        result = np.zeros((len(points), 2))
        for idx, point in enumerate(points):
            result[idx][0], result[idx][1] = lms[point][1], lms[point][2]
        return result
    
    def get_dists(self, lms, points, root=0):
        rootx, rooty = lms[root][1], lms[root][2]
        result = np.zeros(len(points))
        for idx, point in enumerate(points):
            distx = np.abs(point[0] - rootx)
            disty = np.abs(point[1] - rooty)
            result[idx] = m.hypot(distx, disty)
        return result

    def find_landmarks(self, img):
        lms = []
        blank = np.zeros(img.shape)
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            for id, lm in enumerate(hand.landmark):
                h, w = img.shape[:2]
                x, y = int(lm.x * w), int(lm.y * h)
                lms.append((id, x, y))
                if id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20]:
                    cv2.circle(blank, (x, y), 5, (255, 0, 255), cv2.FILLED)
        return (lms, blank)


class Create:
    def __init__(self, cam, size, user) -> None:
        self.ht = HandTracking()
        self.user = user
        self.cam = cam
        self.size = size
        self.fps = self.cam.get(cv2.CAP_PROP_FPS)

    # L E enter(ok) d r
    def create_data(self, element, start=0) -> None:
        # Reset variables.
        num_frames = 0
        num_imgs_taken = 0
        seconds = 0

        # Data collection loop.
        while True:
            seconds = int(num_frames / self.fps)
            _, frame = self.cam.read()
            # flipping the frame to prevent inverted image of captured frame...
            frame = cv2.flip(frame, 1)
            frame_copy = frame.copy()

            if seconds <= 1:
                cv2.putText(frame_copy, "Collecting data for " + element + " in 5", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            elif seconds == 2:
                cv2.putText(frame_copy, "Collecting data for " + element + " in 4", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            elif seconds == 3:
                cv2.putText(frame_copy, "Collecting data for " + element + " in 3", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            elif seconds == 4:
                cv2.putText(frame_copy, "Collecting data for " + element + " in 2", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            elif seconds == 5:
                cv2.putText(frame_copy, "Collecting data for " + element + " in 1", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)        
            else:
                # Checking if we are able to detect the hand...
                img = self.ht.find_hands(frame)
                lms, blank = self.ht.find_landmarks(img)
                if len(lms) > 0:
                    hand = np.around(self.ht.get_points(lms, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]))
                    x = 10000
                    y = 10000 
                    w = -1
                    h = -1
                    for pos in hand:
                        if x > 5000 or pos[0] < x:
                            x = int(pos[0]) - 20
                        if y > 5000 or pos[1]< y:
                            y = int(pos[1]) - 20
                        if w < 0 or pos[0] > w:
                            w = int(pos[0]) + 20
                        if h < 0 or pos[1] > h:
                            h = int(pos[1]) + 20

                    # Reforms blank
                    blank = blank[y:h,x:w,:]
                    if 0 not in blank.shape:
                        blank = cv2.resize(blank, (128, 128))

                        if num_imgs_taken <= self.size:
                            cv2.imwrite(r".\\gesture\\train\\"+self.user+"\\"+element+"\\" +
                            str(num_imgs_taken+start) + '.jpg', blank)
                        elif num_imgs_taken <= 40 + self.size:
                            cv2.imwrite(r".\\gesture\\test\\"+self.user+"\\"+element+"\\" +
                            str(num_imgs_taken-self.size+start) + '.jpg', blank)
                        else:
                            break

                        # Displays the image.
                        cv2.rectangle(img, (x, y), (w,h), (255, 0, 0), 1)
                        cv2.putText(img, 'Hand detected collected ' + str(num_imgs_taken) + ' images.', (25, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        cv2.imshow("Sign Detection", img)
                        num_imgs_taken +=1
                else:
                    cv2.putText(frame_copy, 'No hand detected...', (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    
                    # Dislpays the image.
                    cv2.imshow("Sign Detection", frame_copy)
            
            # Dislpays the image.
            if seconds <= 5:
                cv2.imshow("Sign Detection", frame_copy)

            # increment the number of frames for tracking
            num_frames += 1

            # Closing windows with Esc key...(any other key with ord can be used too.)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        
    def train_data(self, out_size) -> None:
        train_path = r'.\gesture\train\\'+self.user
        test_path = r'.\gesture\test\\'+self.user
        train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size=(128,128), class_mode='categorical', batch_size=10,shuffle=True)
        test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path, target_size=(128,128), class_mode='categorical', batch_size=10, shuffle=True)

        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(128,128,3)))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))
        model.add(Flatten())
        model.add(Dense(32,activation ="relu"))
        model.add(Dense(64,activation ="relu"))
        model.add(Dense(64,activation ="relu"))
        model.add(Dense(out_size,activation ="softmax"))

        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
        model.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0005)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
        model.fit(train_batches, epochs=10, callbacks=[reduce_lr, early_stop], validation_data = test_batches)

        # Once the model is fitted we save the model using model.save()  function.
        model.save('best_model_dataflair3.h5')


class Predict:
    def __init__(self, cam, model_path, word_dict) -> None:
        self.ht = HandTracking()
        self.cam = cam
        self.fps = self.cam.get(cv2.CAP_PROP_FPS)
        self.model = keras.models.load_model(model_path)
        self.word_dict = word_dict

    def predict(self):
        # Resets values
        num_frames = 0

        while True:
            _, frame = self.cam.read()
            frame = cv2.flip(frame, 1)
                
            # Checking if we are able to detect the hand...
            img = self.ht.find_hands(frame)
            lms, blank = self.ht.find_landmarks(img)

            # Gets hand.
            if len(lms) > 0:
                # Output the prediction.
                hand = np.around(self.ht.get_points(lms, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]))
                x = 10000
                y = 10000 
                w = -1
                h = -1
                for pos in hand:
                    if x > 5000 or pos[0] < x:
                        x = int(pos[0]) - 20
                    if y > 5000 or pos[1] < y:
                        y = int(pos[1]) - 20
                    if w < 0 or pos[0] > w:
                        w = int(pos[0]) + 20
                    if h < 0 or pos[1] > h:
                        h = int(pos[1]) + 20

                # The prediction.
                blank = blank[y:h,x:w,:]
                if 0 not in blank.shape:
                    blank = cv2.resize(blank, (128, 128))
                    cv2.imshow("Resized Image", blank)
                    blank = np.reshape(blank,(1,blank.shape[0],blank.shape[1],3))
                    pred = self.model.predict(blank)

                    # Prediction
                    pred = self.model.predict(blank)
                    cv2.rectangle(img, (x, y), (w,h), (255, 0, 0), 1)

                    if np.amax(pred) > 0.95:
                        cv2.putText(img, self.word_dict[np.argmax(pred)], (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    else:
                        cv2.putText(img, 'None', (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                cv2.putText(img, 'None', (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # incrementing the number of frames for tracking
            num_frames += 1
            # Display the frame with segmented hand
            cv2.imshow("Sign Detection", img)
            # Close windows with Esc
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break


if __name__ == '__main__':
    # Initial variables
    word_dict = {}
    args = _parse_args()
    cam = cv2.VideoCapture(args.device)

    # Checks if a model and word dictionary are already present.
    if not (os.path.exists('best_model_dataflair3.h5') and os.path.exists('word_dict')) or args.train:
        # Builds the creator.
        creator = Create(cam, args.sample_size, args.user)

        # Gets train data for all variables.
        for i, ele in enumerate(args.elements):
            # Path for the element.
            path1 = os.path.join(os.getcwd(), 'gesture\\train\\'+args.user)
            path2 = os.path.join(os.getcwd(), 'gesture\\test\\'+args.user)
            if not os.path.isdir(path1):
                os.mkdir(path1)
            if not os.path.isdir(path2):
                os.mkdir(path2)
            if not os.path.isdir(os.path.join(path1, ele)):
                os.mkdir(os.path.join(path1, ele))
            if not os.path.isdir(os.path.join(path2, ele)):
                os.mkdir(os.path.join(path2, ele))
            # Creates data.
            creator.create_data(ele, args.data_start)

            # Adds word to dictionary.
            word_dict[i] = ele

        with open("word_dict", "w", newline='') as new_file:
            # Saves the word dictonary to a csv so it can be reused.
            w = csv.writer(new_file)

            # loop over dictionary keys and values
            for key, val in word_dict.items():
                w.writerow([key, val])

        # Trains the data.
        creator.train_data(len(word_dict))
    elif args.retrain:
        # Builds the creator.
        creator = Create(cam, args.sample_size, args.user)
        word_dict = {}

        # Loads word dictionary.
        with open('word_dict', mode='r') as infile:
            reader = csv.reader(infile)
            for row in reader:
                if row[1] not in args.elements:
                    word_dict[int(row[0])] = row[1]

        last_len = len(word_dict)

        # Gets train data for all variables.
        for i, ele in enumerate(args.elements):
            word_dict[i+last_len] = ele
            # Path for the element.
            path1 = os.path.join(os.getcwd(), 'gesture\\train\\'+args.user)
            path2 = os.path.join(os.getcwd(), 'gesture\\test\\'+args.user)
            if not os.path.isdir(path1):
                os.mkdir(path1)
            if not os.path.isdir(path2):
                os.mkdir(path2)
            if not os.path.isdir(os.path.join(path1, ele)):
                os.mkdir(os.path.join(path1, ele))
            if not os.path.isdir(os.path.join(path2, ele)):
                os.mkdir(os.path.join(path2, ele))
            # Creates data.
            creator.create_data(ele, args.data_start)

        with open("word_dict", "w", newline='') as new_file:
            # Saves the word dictonary to a csv so it can be reused.
            w = csv.writer(new_file)

            # loop over dictionary keys and values
            for key, val in word_dict.items():
                w.writerow([key, val])

        # Trains the data.
        creator.train_data(len(word_dict))
    elif args.train_no:
        # Builds the creator.
        creator = Create(cam, args.sample_size, args.user)
        word_dict = {}

        # Loads word dictionary.
        with open('word_dict', mode='r') as infile:
            reader = csv.reader(infile)
            for row in reader:
                word_dict[int(row[0])] = row[1]

        # Trains the data.
        creator.train_data(len(word_dict))
    else:
        # Loads word dictionary.
        with open('word_dict', mode='r') as infile:
            reader = csv.reader(infile)
            for row in reader:
                word_dict[int(row[0])] = row[1]
        
    # Predicts the value.
    predictor = Predict(cam, os.path.join(os.getcwd(), 'best_model_dataflair3.h5'), word_dict)

    # Run predictions.
    predictor.predict()

    cv2.destroyAllWindows()
    cam.release()