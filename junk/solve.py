import gc
import numpy as np
# from multiprocessing import Pool, Queue
import cv2
import mss
import time
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController, Listener as KeyboardListener
from os import listdir, getcwd
from os.path import isfile, join
import sys
# from memory_profiler import profile

# import ctypes


# PROCESS_PER_MONITOR_DPI_AWARE = 2

# ctypes.windll.shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)

keyboard = KeyboardController()
mouse = MouseController()

# distances = [[-42, 65], [222, 10], [271, 28], [174, -140], [73, -30], [-196, -8], [-178, -39],
#              [-136, 38], [-81, -38], [-31, -66], [-88, -70], [-167, -69], [15, -98], [178, -96], [35, 26]]
# sx, sy = 0, 0
# for x, y in distances:
#     sx += x
#     sy += y
# print(sx, sy)
distances = []


def process_img(screenshot, lower_bound_color, upper_bound_color, is_green_then_dilate=False):
    show_stuff = False
    # Create a mask with the pixel values within the specified range
    lower_bound = np.array(lower_bound_color)
    upper_bound = np.array(upper_bound_color)

    mask = cv2.inRange(screenshot, lower_bound, upper_bound)

    # Apply the mask to the original image to obtain the filtered image
    filtered_img = cv2.bitwise_and(screenshot, screenshot, mask=mask)

    thresh_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGRA2GRAY)
    _, result = cv2.threshold(thresh_img, 5, 255, cv2.THRESH_BINARY)

    # if show_stuff:
    #     cv2.imshow('raw', result)

    # if is_green_then_dilate:
    # apply morphology open with square kernel to remove small white spots
    # apply morphology close with horizontal rectangle kernel to fill horizontal gap
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 30))
    # morph2 = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    # # kernel2 = np.ones((3, 3), np.uint8)
    # # result = cv2.erode(result, kernel2, iterations=1)

    # cv2.imshow('morph2', morph2)

    # remove small white noise

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    # morph2 = cv2.morphologyEx(morph1, cv2.MORPH_CLOSE, kernel)

    # if show_stuff:
    #     cv2.imshow('morph2', morph2)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # morph1 = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    # if show_stuff:
    #     cv2.imshow('removed small white noise', morph1)

    # # close gaps between potions while trying not to close with random white noise
    # # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    # morph2 = cv2.morphologyEx(morph1, cv2.MORPH_CLOSE, kernel)

    # if show_stuff:
    #     cv2.imshow('closed gaps', morph2)

    if not is_green_then_dilate:
        # filter white noise
        # do connected components processing
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            result, None, None, None, 8, cv2.CV_32S)

        # print(centroids)
        # get CC_STAT_AREA component as stats[label, COLUMN]
        areas = stats[1:, cv2.CC_STAT_AREA]

        result = np.zeros((labels.shape), np.uint8)

        for i in range(0, nlabels - 1):
            if areas[i] >= 20:  # keep
                result[labels == i + 1] = 255

    if show_stuff:
        cv2.imshow('connected', result)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    if show_stuff:
        cv2.imshow('closed', result)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    # cv2.imshow('removed noise', result)

    # bounding box around potion
    cnts = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else []

    cscreenx, cscreeny = screenshot.shape[1]//2, screenshot.shape[0]//2
    # boxes = [[100, cscreeny//2, 10, 10], [-100, cscreeny//2, 10, 10]]
    boxes = []

    cscreenx, cscreeny = screenshot.shape[1]//2, screenshot.shape[0]//2
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # remove small black dots
        # area = cv2.contourArea(c)
        # hull = cv2.convexHull(c)
        # hull_area = cv2.contourArea(hull)
        # solidity = 0 if hull_area == 0 else float(area) / hull_area

        # if solidity < 0.95 and (w > 10 and h > 10):
        if w > 8 and h > 8:  # fov: 95
            # if w > 20 and h > 20:
            if is_green_then_dilate and not (w >= 30 and h >= 30):
                continue
            # cv2.rectangle(screenshot, (x, y), (x + w, y + h), color, 2)
            cx, cy = x+w//2, y+h//2
            dx, dy = cx-cscreenx, cy-cscreeny

            boxes.append([dx, dy, w, h])

    if is_green_then_dilate:
        return boxes

    # sort based on distance from centre of screen to box so that we will always move towards the closest box
    boxes = remove_shot_at_targets_from_selection(boxes)

    color = (40, 50, 255, 255)

    for dx, dy, w, h in boxes:
        x, y = dx+cscreenx-w//2, dy+cscreeny-h//2
        cv2.rectangle(screenshot, (x, y), (x + w, y + h), color, 2)

    color = (10, 10, 255, 255)
    cv2.circle(screenshot, (cscreenx, cscreeny), 30, color, 2)
    # cv2.circle(screenshot, (cscreenx, cscreeny), 10, color, 2)
    cv2.circle(screenshot, (cscreenx, cscreeny), 1, color, 2)
    boxes.sort(key=lambda x: x[0]**2+x[1]**2)
    return boxes


def manual_calibration(given_file_name=None):
    mypath = getcwd() + '\screenshots\pov95'

    files = []
    if given_file_name:
        files.append(join(mypath, given_file_name))
    else:
        for f in listdir(mypath):
            filepath = join(mypath, f)
            if isfile(filepath):
                files.append(filepath)

    for filepath in files:
        screenshot = cv2.imread(filepath)
        screenshot = screenshot[300:1080-300+1]
        screenshot = screenshot[:, 500:1920-500+1]
        cv2.namedWindow('ss')

        cx, cy = screenshot.shape[1]//2, screenshot.shape[0]//2

        # for dist in distances:
        # 	screenshot = cv2.circle(screenshot, (dist[0]+cx, dist[1]+cy), 0, (12, 12, 12), 2)

        def add_point(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                # pixel = screenshot[y,x]
                # print(pixel)
                img_with_added_point = cv2.circle(
                    screenshot, (x, y), 4, (0, 0, 255), 2)
                distances.append([x-cx, y-cy])
                print()
                pixel = screenshot[y, x]
                print('pixel color: ', pixel)
                print(distances)
                return img_with_added_point

        cv2.setMouseCallback('ss', add_point)
        screenshot = cv2.circle(screenshot, (cx, cy), 4, (255, 0, 0), 2)

        boxes = process_img(screenshot, [0, 0, 0], [20, 20, 20])
        # boxes = process_img(screenshot, [69, 63, 64], [87, 87, 76])
        # boxes = process_img(screenshot, [100, 100, 100], [130, 130, 130])
        while True:
            cv2.imshow('ss', screenshot)

            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break


# manual_calibration('quake_pro.png')
# manual_calibration('2023-04-11_20.35.35.png')
# manual_calibration('2023-04-12_20.08.29.png')
# manual_calibration('2023-04-15_12.52.12.png')
# manual_calibration('2023-04-15_15.20.17.png')
# manual_calibration()


shots_at = []  # [[time shot, cum_mousex that it takes to move to where the shot was taken from current position, cum_mousey]]
have_touched_box = 0
see_nothing = 0
# movementx_from_centre, movementy_from_centre = 0, 0
# max_possible_travel_distance_from_centre = 600
# centre = [-movementx_from_centre, -movementy_from_centre, 0, 0]
# print('moved to centre', movementx_from_centre, movementy_from_centre)
# print('moved to centre')

# just realised this see nothing cnt is useless because it shldnt be moving at all if it sees nothing


# camera updates about once every two times the code can complete

# to find centre, can only rely on past few shots, cannot rely on sum of movements
# basically memory will store all x distances to get to past shots from current position
# memory = deque([], maxlen=100)

old_green_box = None
cum_mx, cum_my = 0, 0


def take_action(boxes, green_box):
    if len(boxes) == 0:
        global cum_mx, cum_my, see_nothing
        # use green_leaves at the top left of pit to calibrate back to centre
        ideal_dx, ideal_dy = -280, -133
        # ideal_dx, ideal_dy = -470, -200
        if not green_box:
            cum_dx, cum_dy = moused_to_screend([cum_mx, cum_my])
            return move_to_target([cum_dx, cum_dy, 0, 0])
        dx, dy, w, h = green_box
        # print('moved to green box')
        return move_to_target([dx-ideal_dx, dy-ideal_dy, 0, 0], True)

    global have_touched_box

    target = boxes[0]
    screen_dx, screen_dy, w, h = target
    if (abs(screen_dx) <= np.ceil(w/4) and abs(screen_dy) <= np.ceil(h/4)):
        have_touched_box += 1
        if have_touched_box >= 3:  # 5 is good for fov 95
            # print('boxes len', len(boxes))
            # print('shot')
            have_touched_box = 0
            # add current position to shots_at
            # print("shot")
            mouse.press(Button.right)
            mouse.release(Button.right)

            # THIS NEEDS TO BE PLACED AFTER THE MOUSE SHOOTING??? nvm it doesnt
            shots_at.append([time.time(), screen_dx, screen_dy])

            return
            # return move_to_target([0, 0, 0, 0])
    return move_to_target(target)


def remove_shot_at_targets_from_selection(boxes):
    screen_distance_threshold = 30
    delta_time_threshold = 1  # seconds
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
            # if target has been shot at before
            if (dx-edx)**2+(dy-edy)**2 <= screen_distance_threshold**2:
                should_add = 0
                break
        if should_add:
            res.append(target)
    # print('before:', boxes)
    # print('after:', res)
    return res

# tried to make player move back to origin but accumulation of small inaccuracies causes it fail
# tried to increase fov to quake pro to shoot potions on opposite sides of the pit but causes potion far away to be too small to detect


# k = 545  # fov: 95, theoretical is about 380 for all fov, actual 545 around there works best

# 280 dx, 65.5 deg to 95.2 deg --> 29.7deg change k = 490 theoretical


# k = 490 doesn't work in practice


# possible improvement is to investigate camera and mouse threading issue
# k = 2800
# k = 1000
# k = 400
# k = 380
mouse_pixel_to_degree = 0.15


# use integers only because mouse movement only works in pixels --> integers
def screend_to_moused(screend, k=545):
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


def move_to_target(target, is_moving_to_leaves=False):  # returns realised movement?
    dx, dy, w, h = target

    mousex, mousey = screend_to_moused([dx, dy])

    # set whereever the centre of the screen at as the origin only add realised movement
    # print('new pos', mouse.position)
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


# @profile
def on_press(key):
    global initialised
    if key == Key.f4:
        print('esc')
        initialised = 0
        sys.exit(0)
    if key == Key.f6:  # KeyCode.from_char('a'):
        initialised = -initialised
        if initialised == 1:
            print('initiated')
        else:
            print('stopped')


initialised = -1

# intialize global variables for the pool processes:

cnt = 0


# @profile
def main_solution():
    global cnt
    with mss.mss() as sct:
        mon2 = sct.monitors[-1]

        while True:
            if initialised == 0:
                break
            elif initialised == -1:
                continue

            # it can somehow still be moving occasionally even though its at centre so i guess this doesn't work

            if not check_minecraft_knows_mouse_is_at_centre():
                continue

            # print('passed')

            cum_dx, cum_dy = moused_to_screend([cum_mx, cum_my])
            # print(cum_mx, cum_my)
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
            # screenshot = np.flip(screenshot[:, :, :3], 2)
            # screenshot = np.array(im, dtype=np.uint8)

            green_boxes = process_img(screenshot, [4, 20, 4, 0], [
                                      10, 38, 12, 255], True)
            green_box = green_boxes[0] if len(green_boxes) > 0 else []
            boxes = process_img(screenshot, [0, 0, 0, 0], [20, 20, 20, 255])

            cv2.imshow('screenshot', screenshot)
            action = take_action(boxes, green_box)

            if cv2.waitKey(1) & 0xff == ord('q'):
                break
            collected = gc.collect()


# required for Windows:
if __name__ == "__main__":
    keyboard_listener = KeyboardListener(on_press=on_press)

    keyboard_listener.start()

    main_solution()

    keyboard_listener.join()

    print("program ended")
