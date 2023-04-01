import cv2 as cv
import numpy as np
import os
from time import time
from time import time_ns
import win32gui
from PIL import ImageGrab
from pynput.mouse import Listener as MouseListener
from pynput.keyboard import Listener as KeyboardListener
from pynput.keyboard import Key
import uuid6
import yaml

global input
input = {"mouse":[0,0], "click":["",""], "keyboard":[]}

def on_press(key):
    global input

    key = str(listener2.canonical(key)).strip("'")

    if key not in input['keyboard']:
        input['keyboard'].append(key)
def on_release(key):
    global input
    
    key = str(listener2.canonical(key)).strip("'")
    
    if key in input['keyboard']:
        input['keyboard'].remove(key)

def on_move(x, y):
    global input
    input["mouse"] = [x, y]
    pass

def on_click(x, y, button, pressed):
    global input
    if pressed:
        input["click"] = ["{0}".format(button), '1']
    else:
        input["click"] = ["", '0']


def on_scroll(x, y, dx, dy):
    if x<100 and y<100:
        return False

listener = MouseListener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
listener.start()
listener2 = KeyboardListener(on_press=on_press, on_release=on_release)
listener2.start()

path = os.path.realpath(__file__)
main_dir = os.path.dirname(path)
os.chdir(main_dir)
os.chdir('Data')

with open('data_config.yaml', 'r') as f:
    data_config = yaml.load(f, Loader=yaml.FullLoader)

session = data_config['Data']['session_count']

session += 1

data_config['Data']['session_count'] = session
data_config['Data']['available_session'].append(session)

with open('data_config.yaml', 'w') as f:
    yaml.dump(data_config, f)

os.mkdir(f'session_{session}')
os.chdir(f'session_{session}')


loop_time = time()
screenshots = []

frame_count = 0

while(True):
    frame_count += 1
    hwnd = win32gui.FindWindow(None, 'Overwatch')
    rect = win32gui.GetWindowRect(hwnd)
    screenshot = ImageGrab.grab(bbox=(rect[0], rect[1], rect[2], rect[3]))
    screenshot_name = frame_count

    screenshot.save(f'{screenshot_name}.jpeg')
    # screenshots.append((screenshot_name, screenshot))

    with open('data.txt', '+a', encoding='utf-8') as file:
        file.write(f'{screenshot_name}\t{input}\n')
    
    # print(input['keyboard'])
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()
    """
    if len(screenshots) > 100:
        for screenshot_name, screenshot in screenshots:
            screenshot.save(f'{screenshot_name}.jpeg')
        screenshots = []
    """
    """
    screenshot = np.array(screenshot)
    screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2BGR)

    cv.imshow('Computer Vision', screenshot)

    if len(screenshots) > 100:
        for screenshot_name, screenshot in screenshots:
            screenshot.save(f'{screenshot_name}.jpeg')
        screenshots = []

    #qprint('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()
    if cv.waitKey(1) == ord('q'):
        break
    """
"""
for screenshot_name, screenshot in screenshots:
    screenshot.save(f'{screenshot_name}.jpeg')
"""
# cv.destroyAllWindows()