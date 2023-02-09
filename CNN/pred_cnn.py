from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf
import numpy as np
import json
import os
import pandas as pd


dataset_folder_name = 'CNN/Screen_Capturing/Data'
TRAIN_TEST_SPLIT = 0.8
IM_WIDTH = 198
IM_HEIGHT = 198
IMAGE_SAMPLE_SIZE = 7800

# Setup Dataset
replace_dict = {"'mouse'":'"mouse"',"'click'":'"click"',"'keyboard'":'"keyboard"',"['":'["',"', '":'", "', "']":'"]'}

def clean_input(input):
    for key, value in replace_dict.items():
        input = input.replace(key, value)
    return input

def parse_dataset():
    with open(os.path.join(dataset_folder_name,'data.txt'), 'r', encoding="utf-8") as file:
        files_data = file.read().split("\n")
        
    records = []
    for file_data in files_data[: min(IMAGE_SAMPLE_SIZE, len(files_data)-1)]:
        file_name, inputs = file_data.split("\t")
        inputs = clean_input(inputs)
        inputs = json.loads(inputs)
        file_name = file_name + ""
        data = inputs["mouse"][0], inputs["mouse"][1], inputs["click"][0], inputs["keyboard"], file_name
        records.append(data)

    return records

def pd_dataset(dataset):
    df = pd.DataFrame(dataset)
    df.columns = ['x', 'y', 'click', 'keyboard', 'file']

    return df

dataset = parse_dataset()
df = pd_dataset(dataset)

# Clean Dataset

valid_keys_list = ['w','s','a','d','shift','e','q','1','2','r']
valid_click_list = ["Button.left","Button.right"]

data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def clean_data(df):
    
    records = []

    for index, key in df.iterrows():
        data = [0] * 10
        keyboard_keys = str(key['keyboard'])
        keyboard_keys = keyboard_keys.replace("'", '"')
        keyboard_keys = json.loads(keyboard_keys)

        data.insert(0, int(key['x']))
        data.insert(1, int(key['y']))

        for i, k in enumerate(valid_keys_list):
            if k in keyboard_keys:
                data[i+2] = 1

        if valid_click_list[0] in key['click']:
            data.insert(2, 1)
        else:
            data.insert(2, 0)

        if valid_click_list[1] in key['click']:
            data.insert(3, 1)
        else:
            data.insert(3, 0)

        data.append(key['file'])

        records.append(data)
    return records

def pd_labels(dataset):
    df = pd.DataFrame(dataset)
    df.columns = ['x','y','left_click','right_click','w','s','a','d','shift','e','q','1','2','r','file']

    return df

dataset = clean_data(df)
df = pd_labels(dataset)

max_x = df['x'].max()
max_y = df['y'].max()



model = tf.keras.models.load_model('CNN/model.h5')

def pred_frame(img):
    img = img.resize((IM_WIDTH, IM_HEIGHT))
    img = np.array(img) / 255.0
    images = []
    images.append(img)

    images = np.array(images)

    x_pred, y_pred, left_click_pred, right_click_pred, w_pred, s_pred, a_pred, d_pred, shift_pred, e_pred, q_pred, one_pred, two_pred, r_pred = model.predict(images, verbose=0)

    left_click_pred, right_click_pred, w_pred, s_pred, a_pred, d_pred, shift_pred, e_pred, q_pred, one_pred, two_pred, r_pred = left_click_pred.argmax(axis=-1), right_click_pred.argmax(axis=-1), w_pred.argmax(axis=-1), s_pred.argmax(axis=-1), a_pred.argmax(axis=-1), d_pred.argmax(axis=-1), shift_pred.argmax(axis=-1), e_pred.argmax(axis=-1), q_pred.argmax(axis=-1), one_pred.argmax(axis=-1), two_pred.argmax(axis=-1), r_pred.argmax(axis=-1)
    x_pred, y_pred = x_pred * max_x, y_pred * max_y

    return x_pred, y_pred, left_click_pred, right_click_pred, w_pred, s_pred, a_pred, d_pred, shift_pred, e_pred, q_pred, one_pred, two_pred, r_pred

import win32gui
from PIL import ImageGrab
from time import time
import cv2 as cv

loop_time = time()
while(True):
    hwnd = win32gui.FindWindow(None, '(21) room tour 2023 - Youtube - Google Chrome')
    rect = win32gui.GetWindowRect(hwnd)
    screenshot = ImageGrab.grab(bbox=(rect[0], rect[1], rect[2], rect[3]))

    """screenshot = np.array(screenshot)
    screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2BGR)"""

    print(pred_frame(screenshot))

    screenshot = np.array(screenshot)
    screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2BGR)

    cv.imshow('Computer Vision', screenshot)

    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()
    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()