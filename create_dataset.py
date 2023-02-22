import time
import keyboard
import uuid
import logging
import os
import pyautogui
import pyscreeze

DATASET_FOLDER = "dataset"


if __name__ == "__main__":
    logging.info("starting up screenshot loop")
    if not os.path.isdir(f"{DATASET_FOLDER}/"):
        os.mkdir(DATASET_FOLDER)

    time.sleep(3)
    x, y = pyscreeze.locateCenterOnScreen(image="top_left.png")
    bottom_right = pyscreeze.locateCenterOnScreen(image="bottom_right.png")

    print(f"detected game's top left corner at {x,y, bottom_right}")
    # region = (x, y, 800, 1000)
    #
    # while True:
    #     time.sleep(1./30.)
    #     # event = keyboard.is_pressed()
    #     if keyboard.is_pressed('space'):
    #         pyautogui.screenshot(f'{DATASET_FOLDER}/{uuid.uuid4()}.png', region=region)
    #     else:
    #         pyautogui.screenshot(f'{DATASET_FOLDER}/{uuid.uuid4()}.png', region=region)
