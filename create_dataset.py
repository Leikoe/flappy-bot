import time
import pyautogui
import keyboard


def take_screen(i: int, path: str):
    pyautogui.screenshot(path, region=(0, 125, 800, 1200))


def main():
    jump = 0
    no_jump = 0
    time.sleep(7)
    while True:
        time.sleep(0.03)
        if keyboard.is_pressed('space'):
            take_screen(jump, f'dataset/jump/jump_screenshot{jump}.png')
            jump += 1
            print('Added screenshot in jump directory')
        else:
            take_screen(
                no_jump, f'dataset/no_jump/no_jump_screenshot{no_jump}.png')
            no_jump += 1
            print('Added screenshot in no jump directory')


if __name__ == "__main__":
    main()
