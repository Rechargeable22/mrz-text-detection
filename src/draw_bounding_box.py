from pathlib import Path

import imutils
import numpy as np
import tensorflow as tf
import cv2 as cv
from tensorflow import keras
from matplotlib import pyplot as plt
import matplotlib.path as mpltPath
from sklearn.cluster import DBSCAN
from sklearn import metrics


from src.image_loading import validation_crop


def draw_bb_sample_image(image_path="../rp_512.jpg"):
    image_path = Path(Path(__file__).parent / image_path).resolve().__str__()
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (512, 512), preserve_aspect_ratio=True)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize_with_crop_or_pad(image, 512, 512)
    image_gray = tf.image.rgb_to_grayscale(image)
    image_gray = np.stack([image_gray], axis=0)
    image = image.numpy()

    model = keras.models.load_model(Path(__file__).parent / "../data/trained_models/model_shallow_flexible_input.h5")

    predictions = model(image_gray).numpy()

    mrz_lines_image, thresh_pred, contour_image = extract_mrz_lines_bb(image, image_gray[0], predictions[0],
                                                                       return_steps=True)

    fig, axs = plt.subplots(1, 4, figsize=(14, 10))
    axs[0].imshow(image)
    axs[0].set_title('input image')
    axs[1].imshow(thresh_pred, cmap="gray")
    axs[1].set_title('threshold model predictions')
    axs[2].imshow(contour_image)
    axs[2].set_title('10 largest contours found')
    axs[3].imshow(mrz_lines_image)
    axs[3].set_title('contours with most votes')
    fig.tight_layout()
    plt.show()


def draw_dbscan_clustering(number=3, model_name="model_shallow_flexible_input"):
    fig, axs = plt.subplots(number, 3, figsize=(14, 10))
    samples, samples_gray, predictions = generate_samples_and_predictions(number, 512, 512, model_name)
    for idx in range(len(samples)):
        mrz_lines_image, thresh_pred, contour_image = extract_mrz_lines_bb(samples[idx], samples_gray[idx], predictions[idx], return_steps=True)

        prediction_copy = predictions[idx, :, :, 0]
        prediction_copy = 255 * prediction_copy
        prediction_copy = prediction_copy.astype(np.uint8)
        # cv.threshold(prediction_copy, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU, prediction_copy)
        cv.threshold(prediction_copy, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU, prediction_copy)
        contours, hier = cv.findContours(prediction_copy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        centers = []
        for c in contours:
            # get the bounding rect
            x, y, w, h = cv.boundingRect(c)
            # draw a white rectangle to visualize the bounding rect
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            cv.rectangle(prediction_copy, (x, y), (x + w, y + h), 255, 1)
            cv.circle(prediction_copy, (center_x, center_y), radius=5, color=255, thickness=-1)
            centers.append([center_x, center_y])

        cv.drawContours(prediction_copy, contours, -1, (255, 255, 0), 1)

        centers = np.array(centers)

        dbscan = DBSCAN(eps=15)
        # dbscan.fit(centers[:, 0].reshape(-1,1), centers[:, 1])
        dbscan.fit(centers)
        labels = dbscan.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)

        unique_labels = set(labels)
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[dbscan.core_sample_indices_] = True

        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = centers[class_member_mask & core_samples_mask]
            axs[idx, 2].plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=14,
            )

            xy = centers[class_member_mask & ~core_samples_mask]
            axs[idx, 2].plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,
            )



        axs[idx, 0].imshow(samples[idx])
        axs[idx, 0].set_title('input image')

        axs[idx, 1].imshow(prediction_copy, cmap="gray")
        axs[idx, 1].set_title('threshold model predictions')

        # axs[idx, 2].imshow(contour_image)
        axs[idx, 2].set_title('10 largest contours found')

        # axs[idx, 3].imshow(mrz_lines_image)
        # axs[idx, 3].set_title('contours with most votes')

    fig.tight_layout()
    plt.show()


def draw_mrz_lines_extraction_steps(number=3, model_name="model_shallow_flexible_input"):
    fig, axs = plt.subplots(number, 4, figsize=(14, 10))
    samples, samples_gray, predictions = generate_samples_and_predictions(number, 512, 512, model_name)
    for idx in range(len(samples)):
        mrz_lines_image, thresh_pred, contour_image = extract_mrz_lines_bb(samples[idx], samples_gray[idx],
                                                                           predictions[idx], return_steps=True)
        axs[idx, 0].imshow(samples[idx])
        axs[idx, 0].set_title('input image')

        axs[idx, 1].imshow(thresh_pred, cmap="gray")
        axs[idx, 1].set_title('threshold model predictions')

        axs[idx, 2].imshow(contour_image)
        axs[idx, 2].set_title('10 largest contours found')

        axs[idx, 3].imshow(mrz_lines_image)
        axs[idx, 3].set_title('contours with most votes')

    fig.tight_layout()
    plt.show()


def extract_mrz_lines_bb(image, image_gray, predictions, return_steps=False):
    rect_kernel, _ = get_kernels(image_gray)
    gray = cv.GaussianBlur(image_gray, (5, 5), 0)
    blackhat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, rect_kernel)  # reveal dark regions

    gradX = cv.Sobel(blackhat, ddepth=cv.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
    gradX = cv.morphologyEx(gradX, cv.MORPH_CLOSE, rect_kernel)

    thresh = cv.threshold(gradX, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    # thresh = cv.erode(thresh, None, iterations=1)
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:10]  # only search 10 largest contours

    # draw 10 largest contours
    contour_image = image.copy()
    for c in cnts:
        min_rect = cv.minAreaRect(c)
        box = cv.boxPoints(min_rect)
        box = np.intp(box)
        cv.drawContours(contour_image, [box], 0, 0, 2)

    # thresh segmentation mask
    predictions = predictions * 255
    predictions = predictions.astype("uint8")
    thresh_pred = cv.threshold(predictions[:, :, 0], 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    angle_bracket_pixels = cv.findNonZero(thresh_pred)
    angle_bracket_pixels = angle_bracket_pixels.reshape(-1, 2)

    contour_hits = []
    for contour in cnts:
        min_rect = cv.minAreaRect(contour)
        box = cv.boxPoints(min_rect)
        box = np.intp(box)
        box_path = mpltPath.Path(box.tolist())
        pixels_inside = box_path.contains_points(angle_bracket_pixels)
        contour_hits.append((box, pixels_inside.sum()))

    boxes_with_most_angle_brackets = sorted(contour_hits, key=lambda x: x[1], reverse=True)
    boxes_with_most_angle_brackets = np.array(boxes_with_most_angle_brackets)[:2, 0]

    mrz_lines_image = image.copy()
    [cv.drawContours(mrz_lines_image, [box], 0, 0, 2) for box in boxes_with_most_angle_brackets]

    if return_steps:
        return mrz_lines_image, thresh_pred, contour_image
    else:
        return mrz_lines_image


def generate_samples_and_predictions(number, out_height=1024, out_width=1024,
                                     model_name="model_shallow_flexible_input"):
    samples = []
    samples_gray = []
    for _ in range(number):
        sample, sample_gray = validation_crop(out_height, out_width)
        samples.append(sample.numpy())
        samples_gray.append(sample_gray.numpy())

    batched_samples_gray = np.stack(samples_gray, axis=0)
    model = keras.models.load_model(Path(__file__).parent / f"../data/trained_models/{model_name}.h5")
    predictions = model(batched_samples_gray).numpy()
    return samples, samples_gray, predictions


def get_kernels(image):
    # reduce the number of magic numbers by making the kernel size dependent on the image size
    # we assume pictures to be 512x512 or 1024x1024 in size, other sizes can be added
    max_img_size = max(image.shape[0], image.shape[1])
    if max_img_size <= 512:
        rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (13, 5))
        sq_kernel = cv.getStructuringElement(cv.MORPH_RECT, (21, 21))
        return rect_kernel, sq_kernel
    elif max_img_size <= 1024:
        rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (26, 10))
        sq_kernel = cv.getStructuringElement(cv.MORPH_RECT, (42, 42))
        return rect_kernel, sq_kernel
    else:
        raise Exception("Input image size not supported")


if __name__ == '__main__':
    # draw_bb_sample_image("../sample2.jpg")
    # draw_mrz_lines_extraction_steps()
    draw_dbscan_clustering()
