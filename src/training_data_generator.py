import re
import random
from pathlib import Path

import cv2 as cv
import numpy as np
import tensorflow as tf
from scipy.spatial.transform import Rotation as R
from PIL import ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt

from src.image_loading import crop_background


def rgb2gray(img):
    return np.expand_dims(np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]), axis=-1)


def generate_string(characters, columns, rows):
    test_string = (
        "\n".join(
            "".join(
                random.choices(characters, k=columns)
            ) for _ in range(rows)
        )
    )
    return test_string


def draw_string(img, string, font):
    b, g, r, a = 255, 255, 255, 0
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((0, 0), string, font=font, fill=(b, g, r, a))
    return (np.array(img_pil)[..., 0] > 127).astype(np.float32)


def generate_mrz(n_rows, n_cols, font_height, font_width, spacing, font, chars):
    mrz_height = n_rows * (font_height + spacing)
    mrz_width = n_cols * font_width
    img = np.zeros((mrz_height, mrz_width, 3), np.uint8)
    label_img = np.zeros((mrz_height, mrz_width, 3), np.uint8)

    random_string = generate_string(chars, n_cols, n_rows)
    label_random_string = re.sub(re.compile("[^<,\n]"), " ", random_string)
    label_img = draw_string(label_img, label_random_string, font)
    img = draw_string(img, random_string, font)
    return img, label_img


def sample_transform_params(x_min_angle, x_max_angle, y_min_angle, y_max_angle, z_min_angle, z_max_angle,
                            min_width_shift, max_width_shift, min_height_shift, max_height_shift, min_scale, max_scale):
    x_angle = np.random.uniform(x_min_angle, x_max_angle)
    y_angle = np.random.uniform(y_min_angle, y_max_angle)
    z_angle = np.random.uniform(z_min_angle, z_max_angle)
    width_shift = np.random.uniform(min_width_shift, max_width_shift)
    height_shift = np.random.uniform(min_height_shift, max_height_shift)
    scale = np.random.uniform(min_scale, max_scale)
    return x_angle, y_angle, z_angle, width_shift, height_shift, scale


def transform_img(img, target_width, target_height, x_angle, y_angle, z_angle, width_shift, height_shift, scale):
    img_height, img_width = img.shape

    offset = np.array([
        [1.0, 0, -img_width / 2],
        [0, 1, -img_height / 2],
        [0, 0, 1],
    ])

    translate_target_center = np.array([
        [1.0, 0, target_width / 2],
        [0, 1, target_height / 2],
        [0, 0, 1],
    ])

    scale_m = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ])

    shift_m = np.array([
        [1., 0, width_shift],
        [0., 1, height_shift],
        [0, 0, 1]
    ])

    r = R.from_euler('xyz', [x_angle, y_angle, z_angle], degrees=True).as_matrix()

    img_warp = cv.warpPerspective(img, shift_m @ translate_target_center @ scale_m @ r @ offset,
                                  (target_width, target_height), borderValue=0, borderMode=cv.BORDER_CONSTANT,
                                  flags=cv.INTER_NEAREST)
    return img_warp


def combine_background_mrz(background, mrz):
    img = np.minimum(background, (1 - mrz)[..., np.newaxis])
    return img


def add_noise(image, noise_type):
    if noise_type == "gauss_noise":
        row, col, ch = image.shape
        mean = 0
        var = 0.01
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_type == "s&p":
        prob = 0.02
        rnd = np.random.rand(image.shape[0], image.shape[1])
        noisy = image.copy()
        noisy[rnd < prob] = 0
        noisy[rnd > 1 - prob] = 1
        return noisy
    elif noise_type == "blur":
        noisy = cv.GaussianBlur(image, (3, 3), 0)
        noisy = cv.GaussianBlur(noisy, (3, 3), 0)
        noisy = cv.GaussianBlur(noisy, (3, 3), 0)
        return noisy
    elif noise_type == "sharpen":
        return cv.filter2D(image, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))


def sample_generator(gray=True, shape=(512, 512), space_between_chars=4,
                     chars='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<<<<<<<<<<<<<<<<<<<<<<<<<<<                           ',
                     n_rows=60, n_cols=120,
                     x_min_angle=-0.015, x_max_angle=0.015,
                     y_min_angle=-0.01, y_max_angle=0.01,
                     z_min_angle=-15, z_max_angle=15,
                     min_width_shift=-15, max_width_shift=15,
                     min_height_shift=-25, max_height_shift=25,
                     min_scale=0.35, max_scale=0.7):
    sample_height, sample_width = shape

    font = ImageFont.truetype(Path(f"{Path(__file__).parent}/../data/fonts/OCRB.ttf").resolve().__str__(), 40)
    font_height = max(font.getsize(c)[1] for c in chars)
    font_width = max(font.getsize(c)[0] for c in chars)

    while True:
        bg_real = crop_background(sample_height, sample_width, False)

        img, label = generate_mrz(n_rows, n_cols, font_height, font_width, space_between_chars, font, chars)

        x_angle, y_angle, z_angle, width_shift, height_shift, scale = sample_transform_params(x_min_angle, x_max_angle,
                                                                                              y_min_angle, y_max_angle,
                                                                                              z_min_angle, z_max_angle,
                                                                                              min_width_shift,
                                                                                              max_width_shift,
                                                                                              min_height_shift,
                                                                                              max_height_shift,
                                                                                              min_scale, max_scale)

        img_warp = transform_img(img, sample_width, sample_height, x_angle, y_angle, z_angle, width_shift, height_shift,
                                 scale)
        label = transform_img(label, sample_width, sample_height, x_angle, y_angle, z_angle, width_shift, height_shift,
                              scale)

        img = combine_background_mrz(bg_real, img_warp)
        # add augmentations. I could use sophisticated libraries such as imgaug, torchvision, albumentations, augmentor
        # or scikit-image. But for this project I'll stick with opencv
        # further filters to implement: contrast, sharpen, gamma, brightness, jpg compression
        img = add_noise(img, "s&p") if random.random() > 0.5 else img
        img = add_noise(img, "gauss_noise") if random.random() > 0.5 else img
        img = add_noise(img, "blur") if random.random() > 0.75 else img  # blur only every fourth image

        if gray:
            img = rgb2gray(img)

        label = np.concatenate((label[..., np.newaxis], 1 - label[..., np.newaxis]), axis=2)

        yield img, label


def plot_samples(samples=3, shape=(512, 512)):
    ds_gen_test = tf.data.Dataset.from_generator(sample_generator, output_signature=(
        tf.TensorSpec(shape=(*shape, 1), dtype=tf.float32), tf.TensorSpec(shape=(*shape, 2), dtype=tf.float32)))

    fig, ax = plt.subplots(samples, 2, figsize=(12, 12))
    samples_list = list(ds_gen_test.take(samples))
    for ix, sample in enumerate(samples_list):
        img, label = sample
        ax[ix][0].imshow(img, cmap='gray')
        ax[ix][1].imshow(label[..., 0], cmap='gray')
    # plt.savefig("../data/plots/synthetic_input_augmented.png")
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_samples()
