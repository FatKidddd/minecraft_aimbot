import numpy as np
import cv2
import mss
import time
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController, Listener
from os import listdir, getcwd
from os.path import isfile, join
from multiprocessing import Process, Event
from functools import partial
import sys
		 
keyboard = KeyboardController()
mouse = MouseController()

distances = [[-42, 65], [222, 10], [271, 28], [174, -140], [73, -30], [-196, -8], [-178, -39], [-136, 38], [-81, -38], [-31, -66], [-88, -70], [-167, -69], [15, -98], [178, -96], [35, 26]]

def manual_calibration():
	filepath = getcwd() + '\screenshots\photo_2023-04-09_15-02-50.jpg'

	screenshot = cv2.imread(filepath)
	cv2.namedWindow('ss')

	cx, cy = screenshot.shape[1]//2, screenshot.shape[0]//2

	for dist in distances:
		screenshot = cv2.circle(screenshot, (dist[0]+cx, dist[1]+cy), 5, (12, 12, 12), 2)

	def add_point(event,x,y,flags,param):
		if event == cv2.EVENT_LBUTTONDBLCLK:
			# pixel = screenshot[y,x]
			# print(pixel)
			img_with_added_point = cv2.circle(screenshot, (x, y), 5, (0, 0, 255), 2)
			distances.append([x-cx, y-cy])
			print()
			print(distances)
			return img_with_added_point

	cv2.setMouseCallback('ss', add_point)
	lower_bound = np.array([10, 10, 10])
	upper_bound = np.array([17, 17, 17])
	mask = cv2.inRange(screenshot, lower_bound, upper_bound)

	# Apply the mask to the original image to obtain the filtered image
	filtered_img = cv2.bitwise_and(screenshot, screenshot, mask=mask)
	thresh_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
	_,thresh_img = cv2.threshold(thresh_img,5,255,cv2.THRESH_BINARY)

	screenshot = cv2.circle(screenshot, (cx, cy), 5, (255, 0, 0), 2)
	# filter white noise 
	# do connected components processing
	nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img, None, None, None, 8, cv2.CV_32S)
	
	# print(centroids)
	#get CC_STAT_AREA component as stats[label, COLUMN] 
	areas = stats[1:,cv2.CC_STAT_AREA]

	result = np.zeros((labels.shape), np.uint8)

	for i in range(0, nlabels - 1):
		if areas[i] >= 30:   #keep
			result[labels == i + 1] = 255

	# kernel = np.ones((5,5),np.uint8)
	# dilation = cv2.dilate(result,kernel,iterations = 1)
	# bounding box around potion
	cnts = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else []

	for c in cnts:
		x, y, w, h = cv2.boundingRect(c)
		# aspect_ratio = float(w) / h

		area = cv2.contourArea(c)
		# x, y, w, h = cv2.boundingRect(c)
		# rect_area = w * h
		# extent = float(area) / rect_area

		hull = cv2.convexHull(c)
		hull_area = cv2.contourArea(hull)
		solidity = float(area) / hull_area

		# equi_diameter = np.sqrt(4 * area / np.pi)

		# (x, y), (MA, ma), Orientation = cv2.fitEllipse(c)

		# print(" Width = {}  Height = {} area = {}  aspect ration = {}  extent  = {} solidity = {}   equi_diameter = {}   orientation = {}".format(  w , h , area ,aspect_ratio , extent , solidity , equi_diameter , Orientation))
		# x,y,w,h = cv2.boundingRect(c)
		# # remove small black dots
		if solidity < 0.95: #w > 8 and h > 8:
			cv2.rectangle(result, (x, y), (x + w, y + h), (36,255,12), 2)

	while True:
		cv2.imshow('ss', screenshot)

		if cv2.waitKey(1) == ord('q'):
			cv2.destroyAllWindows()
			break


def process_img(screenshot):
	# Create a mask with the pixel values within the specified range
	lower_bound = np.array([10, 10, 10])
	upper_bound = np.array([17, 17, 17])
	mask = cv2.inRange(screenshot, lower_bound, upper_bound)

	# Apply the mask to the original image to obtain the filtered image
	filtered_img = cv2.bitwise_and(screenshot, screenshot, mask=mask)
	thresh_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
	_,thresh_img = cv2.threshold(thresh_img,5,255,cv2.THRESH_BINARY)

	# filter white noise 
	# do connected components processing
	nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img, None, None, None, 8, cv2.CV_32S)
	
	# print(centroids)
	#get CC_STAT_AREA component as stats[label, COLUMN] 
	areas = stats[1:,cv2.CC_STAT_AREA]

	result = np.zeros((labels.shape), np.uint8)

	for i in range(0, nlabels - 1):
			if areas[i] >= 30:   #keep
					result[labels == i + 1] = 255
	
	# bounding box around potion
	cnts = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else []

	cscreenx, cscreeny = screenshot.shape[1]//2, screenshot.shape[0]//2
	# boxes = [[100, cscreeny//2, 10, 10], [-100, cscreeny//2, 10, 10]]
	boxes = []
	for c in cnts:
			x,y,w,h = cv2.boundingRect(c)
			# remove small black dots
			area = cv2.contourArea(c)
			hull = cv2.convexHull(c)
			hull_area = cv2.contourArea(hull)
			solidity = 0 if hull_area == 0 else float(area) / hull_area

			if solidity < 0.95 and (w > 10 and h > 10):
				cv2.rectangle(screenshot, (x, y), (x + w, y + h), (36,255,12), 2)
				cscreenx, cscreeny = screenshot.shape[1]//2, screenshot.shape[0]//2
				cx, cy = x+w//2, y+h//2
				dx, dy = cx-cscreenx, cy-cscreeny

				boxes.append([dx, dy, w, h])

	return boxes

def get_sq_dist(x, y):
	return x**2 + y**2

idx_movements = [] # 0: [mx, my], 1: [mx, my], 2: ...

k = 380

def create_movements_from_distances():
	for dist in distances:
		dx, dy = dist
		scaledx = np.arctan(dx/k)*180/np.pi/0.15
		scaledy = np.arctan(dy/k)*180/np.pi/0.15
		idx_movements.append([scaledx, scaledy])

def auto_calibrate(target):
	dx, dy, w, h = target
	scaledx = np.arctan(dx/k)*180/np.pi/0.15
	scaledy = np.arctan(dy/k)*180/np.pi/0.15
	
	if (abs(dx) <= w/3 and abs(dy) <= h/3):
		mouse.press(Button.right)
		mouse.release(Button.right)
	else:
		mouse.move(scaledx, scaledy)

def move_to_target(target, movement):
	mx, my = movement
	mouse.move(mx, my)
	time.sleep(0.3)
	print('shot')
	mouse.press(Button.right)
	mouse.release(Button.right)
	time.sleep(0.3)
	mouse.move(-mx, -my)
	time.sleep(0.3)

def solve(screenshot):
	for movement in idx_movements:
		mx, my = movement
		mouse.move(mx, my)
		time.sleep(1)
		mouse.move(-mx, -my)
		time.sleep(1)

	# boxes = process_img(screenshot)
	# for box in boxes:
	# 	idx = -1
	# 	min_abs_dist = 1e9
	# 	dx, dy, w, h = box
	# 	camera_dist = get_sq_dist(dx, dy)

	# 	for i, dist in enumerate(distances):
	# 		ax, ay = dist
	# 		actual_dist = get_sq_dist(ax, ay)
	# 		abs_diff = abs(camera_dist-actual_dist) 
	# 		if abs_diff < min_abs_dist:
	# 			min_abs_dist = abs_diff
	# 			idx = i
	# 	assert idx != -1
		
	# 	move_to_target(box, idx_movements[idx])
	# 	break

def thr1(kill_event, flag_event): 
	# manual_calibration()
	create_movements_from_distances()
	print(distances)
	print(idx_movements)

	with mss.mss() as sct:
		while True:
			if kill_event.is_set():
				sys.exit(0)
			if flag_event.is_set():
				mon2 = sct.monitors[2]
				monitor = {
					'top': mon2['top'] + 200,
					'left': mon2['left'] + 120,
					'width': 1920 - 240,
					'height': 1080 - 400,
				}
				screenshot = np.array(sct.grab(monitor))
				screenshot = np.delete(screenshot, -1, axis=2)
				solve(screenshot)
				cv2.imshow('screenshot', screenshot)
				if cv2.waitKey(1) == ord('q'):
					cv2.destroyAllWindows()
					break

def thr2(kill_event, flag_event):
	def on_press(kill_event, flag_event, key):
		if key == Key.f4:
			print('esc')
			kill_event.set()
			sys.exit(0)
		if key == Key.f6: #KeyCode.from_char('a'):
			if flag_event.is_set():
				flag_event.clear()
				print('stopped')
			else:
				flag_event.set()
				print('initiated')

	with Listener(on_press=partial(on_press, kill_event, flag_event)) as listen:
		listen.join()

if __name__ == "__main__":
	kill_event = Event()
	flag_event = Event()

	thread1 = Process(target=thr1, args=(kill_event, flag_event))
	thread2 = Process(target=thr2, args=(kill_event, flag_event))

	thread1.start()
	thread2.start()

	thread1.join()  # Join processes here to avoid main process exit
	thread2.join()