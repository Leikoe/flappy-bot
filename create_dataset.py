import time
import pyautogui
import keyboard


def space_pressed(event) -> bool:
    return event.event_type == keyboard.KEY_DOWN and event.name == 'space'


# i = 0
# while True:
#     pyautogui.screenshot(f'img/test{i}.png')
#     i += 1
#     time.sleep(5)


def main():
    while True:
        event = keyboard.read_event()
        if space_pressed(event):
            print('space was pressed')


if __name__ == "__main__":
    main()
