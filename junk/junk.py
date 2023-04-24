
# if prev_green_box == green_box:
#     continue
# prev_green_box = green_box

# dist = np.sqrt(mx**2+my**2)

# delta_threshold = 0.001*dist
# print((mx, my), 0.001*dist, green_box)

# prev_time = time.time()

# def check_if_should_make_new_movement(prev_d, prev_action, current_d):
#     dx, dy = moused_to_screend(prev_action)
#     prev_dx, prev_dy = prev_d
#     predicted_dx, predicted_dy = prev_dx-dx, prev_dy-dy
#     current_dx, current_dy = current_d
#     predicted_dist = get_mag(predicted_dx, predicted_dy)
#     current_dist = get_mag(current_dx, current_dy)
#     distance_to_cover = abs(predicted_dist-current_dist)
#     return distance_to_cover <= movement_threshold


# prev_d, prev_action = (0, 0), (0, 0)


# mousex, mousey = solve(boxes, green_box)
# # delay here needs to be high enough so that mouse has not moved anymore
# delay = int(np.sqrt(mousex**2+mousey**2))+10
# # print(delay)
# time.sleep(1)

# boxes = process_img(screenshot, [69, 63, 64], [87, 81, 76])
# green_boxes = process_img(screenshot, [8, 31, 9], [10, 38, 12], True)
# orange_boxes = process_img(screenshot, [35,49,67], [48, 67, 94])

# def scale_distance_to_movement(dist, depth):
# 	angle = np.arctan(dist/depth)
# 	movement = angle * 180 / np.pi / 0.15
# 	return movement

# def solve(boxes):
# 	for box in boxes:
# 		dx, dy, w, h = box
# 		mx = scale_distance_to_movement(dx, 300)
# 		my = scale_distance_to_movement(dy, 800)

# 		mouse.move(mx, my)
# 		time.sleep(0.3)
# 		mouse.press(Button.right)
# 		mouse.release(Button.right)
# 		mouse.move(-mx, -my)
# 		time.sleep(0.3)

# dist = get_dist(target)

# dx_flip_sign = -1 if dx < 0 else 1
# dy_flip_sign = -1 if dy < 0 else 1

# scl = 70
# k = 0.05
# xoffset = 80
# yoffset = 0
# dx_log = (scl/(1+np.exp(-k*(abs(dx)-xoffset))) - yoffset) * dx_flip_sign
# dy_log = (scl/(1+np.exp(-k*(abs(dy)-xoffset))) - yoffset) * dy_flip_sign

# scl = 3
# dx_sqrt = np.sqrt(abs(dx)*scl) * dx_flip_sign
# dy_sqrt = np.sqrt(abs(dy)*scl) * dy_flip_sign
# if not timer_event.is_set():
# 	print('right_clicked')
# 	mouse.press(Button.right)
# 	mouse.release(Button.right)
# 	timer_event.set()

# elif dist <= calibration_threshold:
# 	mag = dx_sqrt**2 + dy_sqrt**2
# 	mouse.move(dx_sqrt/mag*5, dy_sqrt/mag*5)
# elif dist >= 100:
# 	mag = np.sqrt(dx**2 + dy**2)
# 	want_mag = 50
# 	mouse.move(dx/mag*want_mag, dy/mag*want_mag)
# elif dist >= 200:
# 	mag = np.sqrt(dx**2 + dy**2)
# 	want_mag = 100
# 	mouse.move(dx/mag*want_mag, dy/mag*want_mag)
# else:
# mouse.move(dx_sqrt, dy_sqrt)
# mouse.move(dx_flip_sign*0.1, dy_flip_sign*0.1)
# mouse.move(dx_log, dy_log)
# mouse.move(target[0]*scl, target[1]*scl)

# val = bucket_map.get(get_idx(dx, dy))
# if val != None:
# 	if not timer_event.is_set():
# 		print('memoised shooting used')
# 		mouse.move(val[0], val[1])
# 		mouse.press(Button.right)
# 		mouse.release(Button.right)
# 		timer_event.set()
# 		mouse.move(-val[0], -val[1])
# 	print('skipped')
# 	return

# global cumx, cumy, original_dx, original_dy

# if (abs(dx) <= w/2 and abs(dy) <= h/2):
# if have_touched_box == 0 or have_touched_box >= 20:
# have_touched_box += 1
# have_touched_box += 1
# if not timer_event.is_set():
# 	timer_event.set()
# print(have_touched_box, 'right_clicked')
# have_touched_box += 1
# if have_touched_box == 1 or have_touched_box >= 10:
# print("shot")
# mouse.press(Button.right)
# mouse.release(Button.right)
# mouse.move(-cumx, -cumy)
# bucket_map[get_idx(original_dx, original_dy)] = (cumx, cumy)
# cumx, cumy = 0, 0
# print(bucket_map)
# time.sleep(0.5)
# print(bucket_map.keys)
# print(k, steps_before_box, prev_steps_before_box)
# if steps_before_box == 1:
# print('ideal k: ', k)
# k += steps_before_box - prev_steps_before_box

# steps_before_box = 0

# print(scaledx, scaledy)
# else:
# have_touched_box = 0
# if cumx == 0 and cumy == 0:
# 	original_dx, original_dy = dx, dy
# cumx += scaledx
# cumy += scaledy
# have_touched_box = 0
# global prev_scaledx, prev_scaledy, jiggle
# print(k)
# same_xdir = scaledx * prev_scaledx > 0
# same_ydir = scaledy * prev_scaledy > 0
# if same_xdir and same_ydir: # same direction
# 	# k -= 1
# 	jiggle = 0
# else:
# 	jiggle += 1
# 	if jiggle >= 3:
# 		mouse.move(10, 10)
# 		jiggle = 0

# prev_scaledx, prev_scaledy = scaledx, scaledy
# steps_before_box += 1


# have_touched_box = 0
# steps_before_box = 0
# prev_steps_before_box = -1
# prev_scaledx, prev_scaledy = 0, 0
# jiggle = 0
# cumx, cumy = 0, 0
# bucket_map = {}

# # def get_idx(x, y):
# # 	rx, ry = int((x//5)*5), int((y//5)*5)
# # 	return str(rx) + '+' + str(ry)

# have_touched_box = 0
# move_cnt = 0

# see_nothing_cnt = 0
# cx, cy = 0, 0
# for shot in memory:
#     cx += shot[0]/len(memory) - 1
#     cy += shot[1]/len(memory) + 3
# move_to_target([cx, cy, 0, 0], timer_event)

# memory.append([0, 0])

# # move to the average of all of last shots so that the player will tend to rotate towards the centre of the pit
# avg_pot_dx, avg_pot_dy = 0, 0
# for shot in shots_at:
#     time_shot, mousex_req, mousey_req = shot
#     avg_pot_dx += mousex_req
#     avg_pot_dy += mousey_req
# imaginary_target = [avg_pot_dx, avg_pot_dy, 0, 0]
# move_to_target(imaginary_target, timer_event)

# next_target = boxes[1] if len(
#     boxes) >= 2 and have_touched_box >= 3 else target
# if have_touched_box >= 3:
#     have_touched_box = 0

# future_dx_from_centre, future_dy_from_centre = moused_to_screend(
#     [movementx_from_centre + mousex, movementy_from_centre + mousey])
# is_within_pit = abs(future_dx_from_centre) <= max_possible_travel_distance_from_centre and abs(
#     future_dy_from_centre) <= max_possible_travel_distance_from_centre

# if not is_within_pit:
#     print('Protected from travelling out of pit')

# if not timer_event.is_set():
# print(mousex, mousey)
# print('moved')

# global movementx_from_centre, movementy_from_centre
# movementx_from_centre += mousex
# movementy_from_centre += mousey
# print(movementx_from_centre, movementy_from_centre)

# for i in range(len(memory)):
#     memory[i][0] += -mousex
#     memory[i][1] += -mousey

# *np.sqrt(dx**2+dy**2)/300)300)
