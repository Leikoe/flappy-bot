import time
import pyautogui
import keyboard


# def space_pressed(event) -> bool:
#     return event.event_type == keyboard.KEY_DOWN and event.name == 'space'


def main():
    jump = 0
    no_jump = 0
    time.sleep(7)
    while True:
        time.sleep(0.03)
        # event = keyboard.is_pressed()
        if keyboard.is_pressed('space'):
            pyautogui.screenshot(f'jump/jump_screenshot{jump}.png')
            jump += 1
            print('Added screenshot in jump directory')
        else:
            pyautogui.screenshot(f'no_jump/no_jump_screenshot{no_jump}.png')
            no_jump += 1
            print('Added screenshot in no jump directory')


if __name__ == "__main__":
    main()
