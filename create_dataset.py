import time
import keyboard
import uuid
import logging
import os
import pyautogui

DATASET_FOLDER = "dataset"


if __name__ == "__main__":
    logging.info("starting up screenshot loop")
    if not os.path.isdir(f"{DATASET_FOLDER}/"):
        os.mkdir(DATASET_FOLDER)

    time.sleep(5)
    while True:
        time.sleep(1./30.)
        # event = keyboard.is_pressed()
        if keyboard.is_pressed('space'):
            pyautogui.screenshot(f'{DATASET_FOLDER}/{uuid.uuid4()}.png')
        else:
            pyautogui.screenshot(f'{DATASET_FOLDER}/{uuid.uuid4()}.png')
