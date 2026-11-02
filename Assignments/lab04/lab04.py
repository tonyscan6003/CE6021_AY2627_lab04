# -*- coding: utf-8 -*-
"""Lab 04 — Watershed Segmentation.

Separate the touching oranges in a box using marker-based Watershed
segmentation. The work is split into small steps, each one a method below
that takes an image (or marker map) in and returns the result out:

  1. colour_channel(image)                 how "orange" each pixel is (red - blue)
  2. threshold(channel)                    oranges -> white, everything else -> black
  3. remove_noise(binary)                  remove small speckle
  4. background_mask(opening)              pixels that are certainly background
  5. distance_transform(opening)           distance of each pixel to the background (or frame)
  6. orange_markers(dist)                  one small marker at each local maximum
  7. unknown_region(bg_mask, marker_mask)  the zone watershed has to decide
  8. create_markers(marker_mask, unknown)  initial labels for watershed
  9. apply_watershed(image, markers)       flood from the markers
 10. segment(image)                        put the steps together

Run evaluate_lab04.py after each step to see your progress.
"""
import cv2
import numpy as np
import os
import sys
from skimage.feature import peak_local_max

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from helpers.dataloader import get_data_path, load_image  # noqa: F401

# Data/Oranges_in_a_box_crop.jpg is a crop (the plastic crate at the bottom
# removed) of "Oranges in a Box" by Sidheeq, CC BY-SA 3.0:
ORANGES_IMAGE_SOURCE_URL = "https://commons.wikimedia.org/wiki/File:Oranges_in_a_Box.jpg"

class WatershedSegmenter:
    """Marker-based Watershed segmentation of touching objects.

    All settings have defaults, so no arguments are needed:

        seg     = WatershedSegmenter()
        markers = seg.segment(image)

    Any setting can be changed with a keyword argument, e.g.:

        seg = WatershedSegmenter(peak_spacing=30)

    Use these settings (self.kernel_size, self.peak_spacing, ...) inside your
    methods rather than typing the numbers in directly.
    """

    def __init__(self, **kwargs):
        self.kernel_size       = kwargs.get('kernel_size',       3)
        self.open_iterations   = kwargs.get('open_iterations',   2)
        self.dilate_iterations = kwargs.get('dilate_iterations', 10)
        self.dist_mask_size    = kwargs.get('dist_mask_size',    5)
        self.peak_spacing      = kwargs.get('peak_spacing',      40)
        self.peak_threshold    = kwargs.get('peak_threshold',    20)
        self.marker_radius     = kwargs.get('marker_radius',     5)

        # Intermediate results, filled in by segment() (step 10)
        self.channel     = None
        self.binary      = None
        self.opening     = None
        self.bg_mask     = None
        self.dist        = None
        self.peaks       = None
        self.marker_mask = None
        self.unknown     = None
        self.markers0    = None
        self.markers     = None

    def colour_channel(self, image):
        """Step 1 — a single-channel image that is bright where the picture is orange.

        Oranges have a lot of red and little blue; the newspaper lining and
        the shadows between the oranges have roughly equal red and blue. So
        red minus blue is large on the oranges and small everywhere else.

        Args:
            image (np.ndarray): H x W x 3 uint8 RGB image.

        Returns:
            np.ndarray: H x W uint8 image, red minus blue (negative values become 0).
        """
        raise NotImplementedError("Implement this method")

    def threshold(self, channel):
        """Step 2 — turn the oranges white and everything else black (Otsu threshold).

        Args:
            channel (np.ndarray): H x W uint8 image from colour_channel().

        Returns:
            np.ndarray: H x W uint8 binary image (values 0 or 255).
        """
        raise NotImplementedError("Implement this method")

    def remove_noise(self, binary):
        """Step 3 — morphological opening to remove small white speckle.

        Args:
            binary (np.ndarray): H x W uint8 binary image from threshold().

        Returns:
            np.ndarray: H x W uint8 binary image, same shape.
        """
        raise NotImplementedError("Implement this method")

    def background_mask(self, opening):
        """Step 4 — the pixels that are certainly background (white).

        Dilating grows the white (orange) regions outward, so whatever is
        still black afterwards is a safe distance from every orange. That
        black region is inverted to give the background mask: white where
        the pixel is certainly background. It becomes the background marker
        in Step 8. Small holes and narrow gaps inside the box are covered by
        the dilation, so they are left for watershed to decide.

        Args:
            opening (np.ndarray): H x W uint8 binary image from remove_noise().

        Returns:
            np.ndarray: H x W uint8 binary image — white = certainly background.
        """
        raise NotImplementedError("Implement this method")

    def distance_transform(self, opening):
        """Step 5 — each white pixel's distance to the nearest black pixel.

        Distances are largest near the centre of each orange, so every orange
        becomes a "hill" with its peak near its centre — even where touching
        oranges have merged into one white region.

        cv2.distanceTransform does not count the image frame as background,
        so for an orange cut off by the frame the distance would keep rising
        right up to the edge. The image is therefore padded with a 1-pixel
        black border first (so the frame counts as background), and the
        padding is cropped off the result.

        Args:
            opening (np.ndarray): H x W uint8 binary image from remove_noise().

        Returns:
            np.ndarray: H x W float32 distance map (in pixels), same shape as opening.
        """
        raise NotImplementedError("Implement this method")

    def orange_markers(self, dist):
        """Step 6 — put one small marker at each local maximum of the distance map.

        Each peak (local maximum) is the centre of one orange. A peak must be
        at least self.peak_spacing pixels from any higher peak, and its
        distance value must be at least self.peak_threshold. A filled circle
        of radius self.marker_radius is drawn at each peak. The peak
        coordinates are also stored on self.peaks.

        Args:
            dist (np.ndarray): H x W distance map from distance_transform().

        Returns:
            np.ndarray: H x W uint8 binary image — white circles at the peaks.
        """
        raise NotImplementedError("Implement this method")

    def unknown_region(self, bg_mask, marker_mask):
        """Step 7 — the zone that is neither background nor a marker —
        this is what watershed will decide.

        Args:
            bg_mask     (np.ndarray): H x W uint8 binary image from background_mask().
            marker_mask (np.ndarray): H x W uint8 binary image from orange_markers().

        Returns:
            np.ndarray: H x W uint8 binary image — white where neither input is white.
        """
        raise NotImplementedError("Implement this method")

    def create_markers(self, marker_mask, unknown):
        """Step 8 — build the initial marker map that watershed floods from.

        Each marker circle gets its own positive label, shifted by +1 so that
        background is 1 (not 0 — cv2.watershed reserves 0 for "unknown").
        The unknown zone is then set back to 0 so watershed knows to decide it.
        Everything left with label 1 is the background mask from Step 4 —
        the background marker.

        Args:
            marker_mask (np.ndarray): H x W uint8 binary image from orange_markers().
            unknown     (np.ndarray): H x W uint8 binary image from unknown_region().

        Returns:
            np.ndarray: H x W int32 marker map (background = 1, unknown = 0,
            each marker a unique integer >= 2).
        """
        raise NotImplementedError("Implement this method")

    def apply_watershed(self, image, markers):
        """Step 9 — flood from the markers over the colour image.

        cv2.watershed uses the colour image's edges (gradients) to decide
        where neighbouring floods meet. It changes the marker array it is
        given, so pass it a copy.

        Args:
            image   (np.ndarray): H x W x 3 uint8 colour image.
            markers (np.ndarray): H x W int32 marker map from create_markers().

        Returns:
            np.ndarray: H x W int32 marker map. Boundary pixels are labelled -1;
            every other pixel has the label of the marker it was flooded from.
        """
        raise NotImplementedError("Implement this method")

    def segment(self, image):
        """Step 10 — run the whole pipeline using the methods above.

        Stores each intermediate result on self (channel, binary, opening,
        bg_mask, dist, peaks, marker_mask, unknown, markers0, markers) so
        they can be displayed afterwards.

        Args:
            image (np.ndarray): H x W x 3 uint8 RGB image.

        Returns:
            np.ndarray: H x W int32 marker map. Boundary pixels are labelled
            -1; background is 1; each detected orange has a unique label >= 2.
        """
        raise NotImplementedError("Implement this method")
