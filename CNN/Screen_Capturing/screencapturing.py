import cv2 as cv
import numpy as np
import os
from time import time
import win32gui
from PIL import ImageGrab
from pynput.mouse import Listener as MouseListener
from pynput.keyboard import Listener as KeyboardListener
from pynput.keyboard import Key
import uuid6

global input
input = {"mouse":[0,0], "click":["",""], "keyboard":[]}

def on_press(key):
    global input
    if hasattr(key, 'char'):
        pass
    else:
        key = key.name

    if key not in input['keyboard']:
        input['keyboard'].append(key)

def on_release(key):
    global input
    if hasattr(key, 'char'):
        pass
    else:
        key = key.name
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


loop_time = time()
while(True):
    hwnd = win32gui.FindWindow(None, 'ARMOURY CRATE')
    rect = win32gui.GetWindowRect(hwnd)
    screenshot = ImageGrab.grab(bbox=(rect[0], rect[1], rect[2], rect[3]))
    screenshot_name = uuid6.uuid6()

    screenshot.save(f'{screenshot_name}.jpeg')

    with open('data.txt', '+a', encoding='utf-8') as file:
        file.write(f'{screenshot_name}\t{input}\n')

    screenshot = np.array(screenshot)
    screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2BGR)

    cv.imshow('Computer Vision', screenshot)

    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()
    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()