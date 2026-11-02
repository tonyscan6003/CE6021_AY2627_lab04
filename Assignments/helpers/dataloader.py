# -*- coding: utf-8 -*-
import cv2
import os
import urllib.request
import numpy as np
from urllib.parse import urlparse

# Absolute path to the shared Data directory (three levels up: helpers/ -> Assignments/ -> repo root)
_here = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(_here)), "Data")


def get_data_path(filename):
    """Return the absolute path to a file inside the shared Data directory."""
    return os.path.join(DATA_DIR, filename)


def download_image(url, save_path):
    """Download an image from a URL into save_path, using a local cache.

    If the file already exists on disk it is read directly without re-downloading.
    Always returns an RGB numpy array.

    Args:
        url (str):       Remote image URL.
        save_path (str): Directory in which to save the image.

    Returns:
        ndarray: H x W x 3 RGB image array.
    """
    filename   = os.path.basename(urlparse(url).path)
    image_path = os.path.join(save_path, filename)

    if os.path.exists(image_path):
        print(f"Image already cached at {image_path}.")
        image = cv2.imread(image_path)
    else:
        print(f"Downloading image from {url} ...")
        os.makedirs(save_path, exist_ok=True)
        resp = urllib.request.urlopen(url)
        temp = np.asarray(bytearray(resp.read()), dtype="uint8")
        temp = cv2.imdecode(temp, cv2.IMREAD_COLOR)
        cv2.imwrite(image_path, temp)          # saved as BGR (cv2 default)
        image = cv2.imread(image_path)
        print(f"Saved to {image_path}.")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_image(image_path, scale_factor=None, as_gray=False):
    """Load an image from disk with optional rescaling and grayscale conversion.

    Args:
        image_path (str):         Absolute path to the image file.
        scale_factor (float):     Divide both spatial dimensions by this factor.
        as_gray (bool):           If True, also return a single-channel greyscale version.

    Returns:
        image (ndarray): H x W x 3 RGB image array.
        gray  (ndarray): H x W greyscale array — only when as_gray=True.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if scale_factor is not None:
        h, w = image.shape[:2]
        image = cv2.resize(
            image,
            dsize=(int(w / scale_factor), int(h / scale_factor)),
            interpolation=cv2.INTER_CUBIC,
        )

    if as_gray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image, gray

    return image


def rescale_crop(image, target_size=512):
    """Rescale image so its shortest side equals target_size, then crop to a square.

    Args:
        image       (ndarray): H x W or H x W x C image array.
        target_size (int):     Side length of the output square in pixels.

    Returns:
        ndarray: target_size x target_size image array.
    """
    img_H, img_W = image.shape[:2]
    scale   = target_size / min(img_H, img_W)
    new_H   = int(np.ceil(img_H * scale))
    new_W   = int(np.ceil(img_W * scale))
    resized = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_AREA)
    return resized[:target_size, :target_size]


def norm_img(img):
    """Normalise an image array to the range [0, 1] as float32.

    Args:
        img (ndarray): Any numeric array.

    Returns:
        ndarray: float32 array with values in [0, 1].
    """
    lo, hi = float(img.min()), float(img.max())
    if hi - lo == 0:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - lo) / (hi - lo)).astype(np.float32)
