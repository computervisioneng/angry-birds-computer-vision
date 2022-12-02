import os
import random

import pymunk as pm
from pymunk import Vec2d
import numpy as np
import cv2
import pyautogui
from sklearn.utils import shuffle
from PIL import ImageChops, Image

from utils import detect_object as detect_object_yolo
from keras_yolo3.yolo import YOLO


YOLO_MODEL = None
YOLO_DIR = os.path.join('.', 'models', 'yolo')
CLEARED_IMG = Image.open('./imgs-util/cleared.png')
FAILED_IMG = Image.open('./imgs-util/failed.png')


def get_yolo_detector(model_weights, confidence_threshold=0.9):

    global YOLO_MODEL

    model_classes = os.path.join(YOLO_DIR, 'classes.names')
    anchors_path = os.path.join(YOLO_DIR, "yolo_anchors.txt")

    if YOLO_MODEL is None:
        YOLO_MODEL = YOLO(
                        **{
                            "model_path": model_weights,
                            "anchors_path": anchors_path,
                            "classes_path": model_classes,
                            "score": confidence_threshold,
                            "gpu_num": 0,
                            "model_image_size": (416, 416),
                        }
                    )

    return YOLO_MODEL


def detect_object(YOLO_MODEL, img):

    img_path = './tmp.png'
    cv2.imwrite(img_path, img)

    prediction, image = detect_object_yolo(
        YOLO_MODEL,
        img_path,
        save_img=False,
        save_img_path=None,
        postfix='',
    )

    os.remove(img_path)

    return prediction


class Bird():
    def __init__(self, distance, angle, x, y, space):
        self.life = 20
        mass = 5
        radius = 12
        inertia = pm.moment_for_circle(mass, 0, radius, (0, 0))
        body = pm.Body(mass, inertia)
        body.position = x, y
        power = distance * 53
        impulse = power * Vec2d(1, 0)
        angle = -angle
        body.apply_impulse_at_local_point(impulse.rotated(angle))
        shape = pm.Circle(body, radius, (0, 0))
        shape.elasticity = 0.95
        shape.friction = 1
        shape.collision_type = 0
        space.add(body, shape)
        self.body = body
        self.shape = shape


def get_distances_array():
    return shuffle([a * 0.5 for a in range(20, 180, 1)])


def get_angles_array():
    return shuffle([l * -0.025 for l in range(10, 40)])


def get_random_shot_coefficients():
    return random.randint(40, 120), -random.randint(25, 70) / 100


def simulate_angle_distance(distance, angle, current_bird_location, target=None):

    x, y = current_bird_location

    x = x - 360
    y = 600 - (y - 240)

    th = 3

    tx, ty = target

    space = pm.Space()
    space.gravity = (0.0, -700.0)

    bird = Bird(distance, angle, x, y, space)

    for j in range(100):
        dt = 1.0 / 50.0 / 2.
        for x in range(2):
            space.step(dt)

        # print([distance, angle, bird.shape.body.position.x, bird.shape.body.position.y])
        # print(np.asarray([bird.shape.body.position.x, bird.shape.body.position.y]))
        if np.linalg.norm(np.asarray([bird.shape.body.position.x, -bird.shape.body.position.y + 600]) -
                          np.asarray([tx, ty])) < th:
            return True, bird.shape.body.position.x, -bird.shape.body.position.y + 600

    return False, 0, 0


def calcdiff(im1, im2):
    dif = ImageChops.difference(im1, im2)
    return np.mean(np.array(dif))


def pass_failed_or_cleared():
    image = pyautogui.screenshot()

    status = np.asarray(image)[324:380, 800:1163, :]

    status = cv2.cvtColor(np.array(status), cv2.COLOR_RGB2BGR)

    cv2.imwrite('./tmp.png', status)

    status = Image.open('./tmp.png')

    th = 10
    if calcdiff(status, CLEARED_IMG) < th:
        pyautogui.moveTo(1045, 770, 2)
        pyautogui.click()
    elif calcdiff(status, FAILED_IMG) < th:
        pyautogui.moveTo(925, 755, 2)
        pyautogui.click()
    else:
        pass  # do nothing
