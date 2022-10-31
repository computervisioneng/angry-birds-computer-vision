import os
import random
import shutil

import cv2
import numpy as np


def get_bbox(mask):

    mask_ = mask.copy()[:, :, :3]
    mask_ = mask_[:, :, 0] + mask_[:, :, 1] + mask_[:, :, 2]

    mask_ = cv2.GaussianBlur(mask_, (3, 3), 0)

    _, mask_ = cv2.threshold(mask_, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask_.copy(), 1, 1)  # not copying here will throw an error
    rect = cv2.minAreaRect(contours[0])  # basically you can feed this rect into your classifier
    (x, y), (w, h), a = rect  # a - angle

    return int(x), int(y), int(w) + 4, int(h) + 4


def overlay_img(background, overlay, location):

    x, y = location

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x > background_width or y > background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
            ],
            axis=2,
        )

    overlay_image = overlay[..., :]
    mask = overlay[..., 3:] / 255.0

    background[y:y + h, x:x + w] = (1.0 - mask) * background[y:y + h, x:x + w] + mask * overlay_image

    return background


def rotate(img, alpha):
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), alpha, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))

    return rotated


background_dir = './imgs/angry-birds-backgrounds/'
pig_img_path = './imgs/pig.png'
bird_img_path = './imgs/red-bird.png'
pause_img_path = './imgs/pause.png'

output_dir = './training_data_with_bird_pause'

train_dir = os.path.join(output_dir, 'train')
imgs_dir = os.path.join(train_dir, 'imgs')
anns_dir = os.path.join(train_dir, 'anns')

for dir_ in [output_dir, train_dir, imgs_dir, anns_dir]:
    if os.path.exists(dir_):
        shutil.rmtree(dir_)
    os.mkdir(dir_)

with open(os.path.join(output_dir, 'classes.names'), 'w') as f:
    f.write('pig\n')
    f.close()

pig_img = cv2.imread(pig_img_path, -1)
bird_img = cv2.imread(bird_img_path, -1)
pause_img = cv2.imread(pause_img_path, -1)

dataset_size = 5000

for j in range(dataset_size):
    alpha = random.randint(0, 359)
    background_path = os.path.join(background_dir,
                                   os.listdir(background_dir)[random.randint(0, len(os.listdir(background_dir)) - 1)])
    background_ = cv2.imread(background_path)
    background_ = cv2.resize(background_, (1200, 650))
    background = np.ones((background_.shape[0], background_.shape[1], 4), dtype=np.uint8) * 255
    background[:, :, :3] = background_

    for i, obj_img in enumerate([pig_img, bird_img, pause_img]):

        resize_ = 50 * random.randint(1, 2)
        pig_img_ = cv2.resize(obj_img, (resize_, resize_))
        pig_img_ = rotate(pig_img_, alpha)

        xc, yc, w, h = get_bbox(pig_img_)

        try:
            location_x, location_y = random.randint(pig_img_.shape[1], background.shape[1] - pig_img_.shape[1]), \
                                        random.randint(pig_img_.shape[0], background.shape[0] - pig_img_.shape[0])
            img_ = overlay_img(background, pig_img_, (location_x, location_y))
            cv2.imwrite(os.path.join(imgs_dir, '{}.jpg'.format(str(j))), img_)

            H, W, _ = img_.shape

            if i == 0:
                with open(os.path.join(anns_dir, '{}.txt'.format(str(j))), 'w') as f:
                    f.write('0 {} {} {} {}\n'.format(str((xc + location_x) / W), str((yc + location_y) / H), str(w / W),
                                                     str(h / H)))
                    f.close()

        except Exception:
            pass
