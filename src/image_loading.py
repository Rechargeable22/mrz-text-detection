# Loads images from the midv500 dataset to use as training background or for visual MRZ extraction validation
import random
from pathlib import Path

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

# Dataset constants
NO_MRZ = ["01_alb_id", "02_aut_drvlic_new", "03_aut_id_old", "04_aut_id",
          "07_chl_id", "08_chn_homereturn", "09_chn_id", "10_cze_id", "12_deu_drvlic_new", "13_deu_drvlic_old"]
MRZ = ["05_aze_passport", "06_bra_passport", "11_cze_passport", "15_deu_id_old", "16_deu_passport_new"]
BACKGROUND = ["CA", "CS", "HA", "HS", "KA", "KS", "PA", "PS", "TA", "TS"]
WIDTH = 1080
HEIGHT = 1920


def rgb2gray(rgb):
    return np.expand_dims(np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]), axis=-1)


def generate_random_image_path(mrz=False):
    id_type = random.choice(MRZ) if mrz else random.choice(NO_MRZ)
    background_type = random.choice(BACKGROUND)
    path = f"{Path(__file__).parent}/../data/midv500_data/midv500/{id_type}/images/" \
           f"{background_type}/{background_type[:2]}{id_type[:2]}_{random.randint(1, 30):02d}.tif"
    return path


def crop_background(out_height, out_width, mrz=False):
    img = cv.imread(generate_random_image_path(mrz))
    crop_width = random.randint(0, WIDTH - out_width)
    crop_height = random.randint(0, HEIGHT - out_height)
    crop_img = img[crop_height:crop_height + out_height, crop_width:crop_width + out_width]
    return crop_img / 255


def show_sample_backgrounds():
    for i in range(5):
        cv.imshow("cropped", crop_background(512, 512))
        cv.waitKey(0)


def validation_crop(out_height, out_width):
    image = cv.imread(generate_random_image_path(True))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    min_height_width = min(image.shape[0], image.shape[1])
    image = tf.image.resize_with_crop_or_pad(image, min_height_width, min_height_width)  # crop/pad center
    image = tf.image.resize(image, (out_height, out_width), preserve_aspect_ratio=True)
    image = tf.cast(image, tf.float32) / 255.0
    image_gray = tf.image.rgb_to_grayscale(image)
    return image, image_gray


def show_validation_crops(height=1024, width=1024):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs[0, 0].imshow(validation_crop(height, width)[0])
    axs[0, 1].imshow(validation_crop(height, width)[0])
    axs[1, 0].imshow(validation_crop(height, width)[1], cmap="gray")
    axs[1, 1].imshow(validation_crop(height, width)[1], cmap="gray")
    fig.tight_layout()
    plt.show()


def sample_image(height=512, width=512):
    image = tf.io.read_file('../sample2.jpg')
    # image = tf.io.read_file('sample2.jpg')
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (height, width), preserve_aspect_ratio=True)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize_with_crop_or_pad(image, height, width)
    image_gray = tf.image.rgb_to_grayscale(image)

    label = tf.io.read_file('../highlighted.jpg')
    # label = tf.io.read_file('highlighted.jpg')
    label = tf.image.decode_jpeg(label, channels=3)
    label = tf.image.resize(label, (height, width), preserve_aspect_ratio=True)
    label = tf.image.rgb_to_grayscale(label)
    label = tf.cast(label, tf.float32) / 255.0
    label = tf.image.resize_with_crop_or_pad(label, height, width)
    label = tf.concat((label, 1 - label), axis=2)
    # label = np.concatenate((label[..., np.newaxis], 1 - label[..., np.newaxis]), axis=2)

    return image, label, image_gray


def sample_image_generator():
    image, label, image_gray = sample_image(512, 512)
    yield image_gray, label


def sample_image_generator_small():
    image, label, image_gray = sample_image(256, 256)
    yield image_gray, label


if __name__ == '__main__':
    show_validation_crops(512, 512)
