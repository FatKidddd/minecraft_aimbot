import gc
import numpy as np
import cv2
import mss
import time
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController, Listener as KeyboardListener
import sys

keyboard = KeyboardController()
mouse = MouseController()

distances = []
shots_at = []
have_touched_box = 0
cum_mx, cum_my = 0, 0
initialised = -1
mouse_pixel_to_degree = 0.15


def process_img(screenshot, lower_bound_color, upper_bound_color, is_green_then_dilate=False):
    lower_bound = np.array(lower_bound_color)
    upper_bound = np.array(upper_bound_color)

    mask = cv2.inRange(screenshot, lower_bound, upper_bound)
    filtered_img = cv2.bitwise_and(screenshot, screenshot, mask=mask)
    thresh_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGRA2GRAY)
    _, result = cv2.threshold(thresh_img, 5, 255, cv2.THRESH_BINARY)

    if not is_green_then_dilate:
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            result, None, None, None, 8, cv2.CV_32S)
        areas = stats[1:, cv2.CC_STAT_AREA]
        result = np.zeros((labels.shape), np.uint8)
        for i in range(0, nlabels - 1):
            if areas[i] >= 20:  # keep
                result[labels == i + 1] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else []

    cscreenx, cscreeny = screenshot.shape[1]//2, screenshot.shape[0]//2
    boxes = []

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)

        if w > 8 and h > 8:  # fov: 95
            if is_green_then_dilate and not (w >= 30 and h >= 30):
                continue
            cx, cy = x+w//2, y+h//2
            dx, dy = cx-cscreenx, cy-cscreeny

            boxes.append([dx, dy, w, h])

    if is_green_then_dilate:
        return boxes

    boxes = remove_shot_at_targets_from_selection(boxes)
    boxes.sort(key=lambda x: x[0]**2+x[1]**2)

    color = (40, 50, 255, 255)

    for dx, dy, w, h in boxes:
        x, y = dx+cscreenx-w//2, dy+cscreeny-h//2
        cv2.rectangle(screenshot, (x, y), (x + w, y + h), color, 2)

    color = (10, 10, 255, 255)
    cv2.circle(screenshot, (cscreenx, cscreeny), 30, color, 2)
    cv2.circle(screenshot, (cscreenx, cscreeny), 1, color, 2)

    return boxes


def take_action(boxes, green_box):
    if len(boxes) == 0:
        global cum_mx, cum_my
        ideal_dx, ideal_dy = -280, -133
        if not green_box:
            cum_dx, cum_dy = moused_to_screend([cum_mx, cum_my])
            return move_to_target([cum_dx, cum_dy, 0, 0])
        dx, dy, w, h = green_box
        return move_to_target([dx-ideal_dx, dy-ideal_dy, 0, 0], True)

    global have_touched_box

    target = boxes[0]
    screen_dx, screen_dy, w, h = target
    if (abs(screen_dx) <= np.ceil(w/4) and abs(screen_dy) <= np.ceil(h/4)):
        have_touched_box += 1
        if have_touched_box >= 3:
            have_touched_box = 0
            mouse.press(Button.right)
            mouse.release(Button.right)
            shots_at.append([time.time(), screen_dx, screen_dy])
            return
    return move_to_target(target)


def remove_shot_at_targets_from_selection(boxes):
    screen_distance_threshold = 30
    delta_time_threshold = 2  # seconds don't ever decrease this value!
    expected_screen_pos = []
    for i in range(len(shots_at)-1, -1, -1):
        time_shot, mousex_req, mousey_req = shots_at[i]
        delta_time = time.time() - time_shot
        if delta_time > delta_time_threshold:
            shots_at.pop(i)
            continue
            # use theoretical k
        delta_screen_dx, delta_screen_dy = moused_to_screend(
            [mousex_req, mousey_req], k=480)
        expected_screen_pos.append([delta_screen_dx, delta_screen_dy])

    res = []
    for target in boxes:
        should_add = 1
        for pos in expected_screen_pos:
            edx, edy = pos
            dx, dy, w, h = target
            if (dx-edx)**2+(dy-edy)**2 <= screen_distance_threshold**2:
                should_add = 0
                break
        if should_add:
            res.append(target)
    return res

# k = 545  # fov: 95, theoretical is about 380 for all fov, actual 545 around there works best
# 280 dx, 65.5 deg to 95.2 deg --> 29.7deg change k = 490 theoretical
# k = 490 doesn't work in practice


def screend_to_moused(screend, k=545):
    # use integers only because mouse movement only works in pixels --> integers
    dx, dy = screend
    scaledx = int(np.arctan(dx/k)*180/np.pi/mouse_pixel_to_degree)
    scaledy = int(np.arctan(dy/k)*180/np.pi/mouse_pixel_to_degree)
    return scaledx, scaledy


def moused_to_screend(moused, k=545):
    dx, dy = moused
    scaledx = int(k*np.tan(dx*mouse_pixel_to_degree*np.pi/180))
    scaledy = int(k*np.tan(dy*mouse_pixel_to_degree*np.pi/180))
    return scaledx, scaledy


def get_mag(x, y):
    return np.sqrt(x**2+y**2)


def move_to_target(target, is_moving_to_leaves=False):
    dx, dy, w, h = target

    mousex, mousey = screend_to_moused([dx, dy])

    for i in range(len(shots_at)):
        shots_at[i][1] += -mousex
        shots_at[i][2] += -mousey

    global cum_mx, cum_my
    cum_mx += -mousex
    cum_my += -mousey

    if is_moving_to_leaves:
        cum_mx, cum_my = 0, 0

    mouse.move(mousex, mousey)

# need to disable enhance pointer precision to remove mouse acceleration


def check_minecraft_knows_mouse_is_at_centre():
    return mouse.position == (-960, 905)


def on_press(key):
    global initialised
    if key == Key.f4:
        print('esc')
        initialised = 0
        sys.exit(0)
    if key == Key.f6:
        initialised = -initialised
        if initialised == 1:
            print('initiated')
        else:
            print('stopped')


def main_solution():
    cnt = 0
    with mss.mss() as sct:
        mon2 = sct.monitors[-1]

        while True:
            if initialised == 0:
                break
            elif initialised == -1:
                continue
            if not check_minecraft_knows_mouse_is_at_centre():
                continue

            cum_dx, cum_dy = moused_to_screend([cum_mx, cum_my])
            margin_top = min(220, int(cum_dy * 2)) if cum_dy >= 0 else 0
            margin_left = 0

            monitor = {
                'top': mon2['top'] + 300 + margin_top,
                'left': mon2['left'] + 500 + margin_left,
                'width': 1920 - 1000 - abs(margin_left) * 2,
                'height': 1080 - 600 - abs(margin_top) * 2,
            }
            im = sct.grab(monitor)
            screenshot = np.array(im, dtype=np.uint8)

            green_boxes = process_img(screenshot, [4, 20, 4, 0], [
                                      10, 38, 12, 255], True)
            green_box = green_boxes[0] if len(green_boxes) > 0 else []
            boxes = process_img(screenshot, [0, 0, 0, 0], [20, 20, 20, 255])

            # cv2.imshow('screenshot', screenshot)
            take_action(boxes, green_box)

            # if cv2.waitKey(1) & 0xff == ord('q'):
            #     break
            cnt += 1
            if cnt >= 500:
                gc.collect()
                cnt = 0


if __name__ == "__main__":
    keyboard_listener = KeyboardListener(on_press=on_press)

    keyboard_listener.start()

    main_solution()

    keyboard_listener.join()

    print("program ended")
