import os
import random
import time
import math

import pyautogui
import numpy as np


from util import detect_object, get_yolo_detector, get_angles_array, get_distances_array, \
    simulate_angle_distance, get_random_shot_coefficients, pass_failed_or_cleared


# angry birds pipeline ######

while True:
    # detect pigs

    model_path = os.path.join('.', 'models', 'yolo', 'yolov3_pig_detector.h5')

    img = pyautogui.screenshot()

    img = np.array(img)

    img = img[239:889, 360:1560, :]

    yolo_detector = get_yolo_detector(model_path)

    detections = detect_object(yolo_detector, img)

    pigs = []
    for detection in detections:
        x1, y1, x2, y2, class_, confidence_value = detection
        xc = int((x1 + x2) / 2)
        yc = int((y1 + y2) / 2)
        pigs.append([xc, yc])

    print(pigs)

    # get current bird

    CURRENT_BIRD_LOCATION = (513, 687)

    pyautogui.moveTo(CURRENT_BIRD_LOCATION[0], CURRENT_BIRD_LOCATION[1], 1)
    pyautogui.mouseDown()

    # aim
    distances_array = get_distances_array()
    angles_array = get_angles_array()

    mouse_distance = None
    selected_angle = None
    for distance in distances_array:
        for angle in angles_array:
            if len(pigs) > 0:
                pig_selected = pigs[random.randint(0, len(pigs) - 1)]
                ret, x, y = simulate_angle_distance(distance, angle, CURRENT_BIRD_LOCATION, pig_selected)

            if ret:
                mouse_distance = distance * 1.1
                selected_angle = angle
                break
        if ret:
            break
    if len(pigs) == 0 or not ret:
        print('RANDOM SHOT')
        mouse_distance, selected_angle = get_random_shot_coefficients()

    print(mouse_distance, selected_angle)

    # shot
    dx = mouse_distance * math.cos(selected_angle)
    dy = mouse_distance * math.sin(selected_angle)

    pyautogui.drag(-dx, -dy - 2, 2)
    pyautogui.mouseUp()

    time.sleep(5)

    pass_failed_or_cleared()

    time.sleep(5)
