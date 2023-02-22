import keyboard
import pyautogui

def select_zone():
    p1, p2 = None, None
    while not p1 or not p2:
        if keyboard.is_pressed('q'):
            p1 = pyautogui.position()
        if keyboard.is_pressed('ctrl+shift+z'):
            p2 = pyautogui.position()
    return p1, p2

if __name__ == '__main__':
    print(select_zone())
