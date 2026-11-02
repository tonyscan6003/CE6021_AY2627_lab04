# -*- coding: utf-8 -*-
"""Tests for Lab 04 — Watershed Segmentation.

Grouped by step — a failing test names the step that needs attention.
"""
import numpy as np
import cv2
import os
import sys
import pytest
from skimage.feature import peak_local_max

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lab04 import WatershedSegmenter
from helpers.dataloader import get_data_path, load_image

@pytest.fixture(scope="module")
def image():
    """The box of oranges (RGB)."""
    return load_image(get_data_path("Oranges_in_a_box_crop.jpg"))

@pytest.fixture(scope="module")
def opening(image):
    seg = WatershedSegmenter()
    return seg.remove_noise(seg.threshold(seg.colour_channel(image)))

def two_touching_discs():
    """A binary image of two overlapping discs (radius 30, centres 50 px apart).

    The canvas is generous because peak_local_max also ignores peaks closer
    to the image border than min_distance.
    """
    img = np.zeros((260, 320), np.uint8)
    cv2.circle(img, (135, 130), 30, 255, -1)
    cv2.circle(img, (185, 130), 30, 255, -1)
    return img

# ── Step 1: colour_channel ────────────────────────────────────────────────

def test_colour_channel_matches_reference(image):
    expected = cv2.subtract(image[:, :, 0], image[:, :, 2])
    assert np.array_equal(WatershedSegmenter().colour_channel(image), expected)

def test_colour_channel_is_red_minus_blue_without_wraparound():
    img = np.zeros((1, 2, 3), np.uint8)
    img[0, 0] = (200, 90, 50)    # orange-ish: red 200, blue 50
    img[0, 1] = (50, 90, 200)    # blue-ish: red < blue
    result = WatershedSegmenter().colour_channel(img)
    assert result.shape == (1, 2)
    assert result[0, 0] == 150
    assert result[0, 1] == 0      # 50 - 200 must give 0, not wrap around to 106

# ── Step 2: threshold ─────────────────────────────────────────────────────

def test_threshold_matches_otsu_reference(image):
    seg = WatershedSegmenter()
    channel = seg.colour_channel(image)
    expected = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    assert np.array_equal(seg.threshold(channel), expected)

def test_threshold_makes_oranges_white():
    channel = np.zeros((10, 10), np.uint8)
    channel[2:8, 2:8] = 200       # a bright "orange" on a dark background
    binary = WatershedSegmenter().threshold(channel)
    assert binary[5, 5] == 255 and binary[0, 0] == 0

# ── Step 3: remove_noise ──────────────────────────────────────────────────

def test_remove_noise_matches_opening_reference(image):
    seg = WatershedSegmenter()
    binary = seg.threshold(seg.colour_channel(image))
    expected = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    assert np.array_equal(seg.remove_noise(binary), expected)

def test_remove_noise_removes_a_single_pixel():
    binary = np.zeros((20, 20), np.uint8)
    binary[10, 10] = 255
    assert np.all(WatershedSegmenter().remove_noise(binary) == 0)

def test_remove_noise_respects_kernel_size(image):
    seg = WatershedSegmenter()
    binary = seg.threshold(seg.colour_channel(image))
    small = WatershedSegmenter(kernel_size=3).remove_noise(binary)
    large = WatershedSegmenter(kernel_size=15).remove_noise(binary)
    assert np.sum(large == 255) < np.sum(small == 255)

def test_remove_noise_respects_open_iterations(image):
    seg = WatershedSegmenter()
    binary = seg.threshold(seg.colour_channel(image))
    few  = WatershedSegmenter(open_iterations=1).remove_noise(binary)
    many = WatershedSegmenter(open_iterations=6).remove_noise(binary)
    assert np.sum(many == 255) < np.sum(few == 255)

# ── Step 4: background_mask ───────────────────────────────────────────────

def test_background_mask_matches_inverted_dilate_reference(opening):
    expected = 255 - cv2.dilate(opening, np.ones((3, 3), np.uint8), iterations=10)
    assert np.array_equal(WatershedSegmenter().background_mask(opening), expected)

def test_background_mask_is_white_away_from_the_object():
    img = np.zeros((60, 60), np.uint8)
    img[25:35, 25:35] = 255
    bg = WatershedSegmenter(dilate_iterations=5).background_mask(img)
    assert bg[0, 0] == 255             # far from the square: background
    assert bg[30, 30] == 0             # on the square: not background
    assert bg[30, 22] == 0             # 3 px outside the square: covered by the dilation
    assert bg[30, 18] == 255           # 7 px outside: beyond the dilation

def test_background_mask_respects_dilate_iterations(opening):
    few  = WatershedSegmenter(dilate_iterations=2).background_mask(opening)
    many = WatershedSegmenter(dilate_iterations=15).background_mask(opening)
    assert np.sum(many == 255) < np.sum(few == 255)

# ── Step 5: distance_transform ────────────────────────────────────────────

def test_distance_transform_matches_padded_reference(opening):
    padded = cv2.copyMakeBorder(opening, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    expected = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    result = WatershedSegmenter().distance_transform(opening)
    assert result.shape == opening.shape
    assert np.allclose(result, expected)

def test_distance_transform_counts_the_frame_as_background():
    """A white region cut off by the frame must peak inside the image, not on the edge."""
    img = np.zeros((40, 60), np.uint8)
    img[0:20, 10:50] = 255             # touches the top edge
    dist = WatershedSegmenter().distance_transform(img)
    assert dist[0, 30] == pytest.approx(1, abs=0.1)   # next to the frame
    assert np.unravel_index(np.argmax(dist), dist.shape)[0] > 0

def test_distance_transform_is_largest_at_disc_centre():
    img = np.zeros((61, 61), np.uint8)
    cv2.circle(img, (30, 30), 25, 255, -1)
    dist = WatershedSegmenter().distance_transform(img)
    assert np.unravel_index(np.argmax(dist), dist.shape) == (30, 30)
    assert dist[0, 0] == 0

def test_distance_transform_respects_dist_mask_size(opening):
    d3 = WatershedSegmenter(dist_mask_size=3).distance_transform(opening)
    d5 = WatershedSegmenter(dist_mask_size=5).distance_transform(opening)
    assert not np.allclose(d3, d5)

# ── Step 6: orange_markers ────────────────────────────────────────────────

def test_orange_markers_matches_peak_local_max_reference(opening):
    seg = WatershedSegmenter()
    dist = seg.distance_transform(opening)
    expected = peak_local_max(dist, min_distance=40, threshold_abs=20)
    seg.orange_markers(dist)
    assert np.array_equal(np.asarray(seg.peaks), expected)

def test_orange_markers_finds_one_marker_per_touching_disc():
    seg = WatershedSegmenter(peak_spacing=10, peak_threshold=5)
    dist = cv2.distanceTransform(two_touching_discs(), cv2.DIST_L2, 5)
    marker_mask = seg.orange_markers(dist)
    n_markers = cv2.connectedComponents(marker_mask)[0] - 1
    assert n_markers == 2

def test_orange_markers_draws_circles_at_peaks():
    seg = WatershedSegmenter(peak_spacing=10, peak_threshold=5, marker_radius=4)
    dist = cv2.distanceTransform(two_touching_discs(), cv2.DIST_L2, 5)
    marker_mask = seg.orange_markers(dist)
    for row, col in seg.peaks:
        assert marker_mask[row, col] == 255        # (row, col) indexing on the array
    assert set(np.unique(marker_mask)).issubset({0, 255})

def test_orange_markers_respects_peak_spacing():
    dist = cv2.distanceTransform(two_touching_discs(), cv2.DIST_L2, 5)
    close = WatershedSegmenter(peak_spacing=10, peak_threshold=5).orange_markers(dist)
    wide  = WatershedSegmenter(peak_spacing=60, peak_threshold=5).orange_markers(dist)
    assert cv2.connectedComponents(close)[0] - 1 == 2
    assert cv2.connectedComponents(wide)[0] - 1 == 1

def test_orange_markers_respects_peak_threshold():
    dist = cv2.distanceTransform(two_touching_discs(), cv2.DIST_L2, 5)
    marker_mask = WatershedSegmenter(peak_spacing=10, peak_threshold=100).orange_markers(dist)
    assert np.all(marker_mask == 0)                # no peak is 100 px from the edge

def test_orange_markers_respects_marker_radius():
    dist = cv2.distanceTransform(two_touching_discs(), cv2.DIST_L2, 5)
    small = WatershedSegmenter(peak_spacing=10, peak_threshold=5, marker_radius=2).orange_markers(dist)
    large = WatershedSegmenter(peak_spacing=10, peak_threshold=5, marker_radius=8).orange_markers(dist)
    assert np.sum(large == 255) > np.sum(small == 255)

# ── Step 7: unknown_region ────────────────────────────────────────────────

def test_unknown_region_is_neither_background_nor_marker():
    bg_mask     = np.array([[0, 0, 255], [0, 0, 255]], dtype=np.uint8)
    marker_mask = np.array([[255, 0, 0], [0, 0, 0]], dtype=np.uint8)
    expected    = np.array([[0, 255, 0], [255, 255, 0]], dtype=np.uint8)
    assert np.array_equal(WatershedSegmenter().unknown_region(bg_mask, marker_mask), expected)

# ── Step 8: create_markers ────────────────────────────────────────────────

def test_create_markers_labels_background_as_one():
    markers = WatershedSegmenter().create_markers(np.zeros((4, 4), np.uint8),
                                                   np.zeros((4, 4), np.uint8))
    assert np.all(markers == 1)

def test_create_markers_sets_unknown_to_zero():
    marker_mask = np.zeros((4, 4), np.uint8); marker_mask[0, 0] = 255
    unknown = np.zeros((4, 4), np.uint8); unknown[3, 3] = 255
    markers = WatershedSegmenter().create_markers(marker_mask, unknown)
    assert markers[3, 3] == 0
    assert markers[0, 0] == 2          # the one marker, shifted by +1

def test_create_markers_gives_each_marker_a_unique_label():
    marker_mask = np.zeros((6, 6), np.uint8); marker_mask[0, 0] = 255; marker_mask[5, 5] = 255
    markers = WatershedSegmenter().create_markers(marker_mask, np.zeros((6, 6), np.uint8))
    assert markers[0, 0] != markers[5, 5]
    assert markers[0, 0] >= 2 and markers[5, 5] >= 2

# ── Step 9: apply_watershed ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def _markers0_base(image):
    seg = WatershedSegmenter()
    op = seg.remove_noise(seg.threshold(seg.colour_channel(image)))
    fg = seg.orange_markers(seg.distance_transform(op))
    return seg.create_markers(fg, seg.unknown_region(seg.background_mask(op), fg))

@pytest.fixture
def markers0(_markers0_base):
    """A fresh copy for every test, so one test can't overwrite another's input."""
    return _markers0_base.copy()

def test_apply_watershed_matches_cv2_reference(image, markers0):
    expected = cv2.watershed(image, markers0.copy())
    assert np.array_equal(WatershedSegmenter().apply_watershed(image, markers0), expected)

def test_apply_watershed_produces_boundaries(image, markers0):
    assert np.any(WatershedSegmenter().apply_watershed(image, markers0) == -1)

def test_apply_watershed_does_not_change_its_input(image, markers0):
    before = markers0.copy()
    WatershedSegmenter().apply_watershed(image, markers0)
    assert np.array_equal(markers0, before)

# ── Step 10: segment (end-to-end) ─────────────────────────────────────────

@pytest.fixture(scope="module")
def segmented(image):
    seg = WatershedSegmenter()
    return seg, seg.segment(image)

def test_segment_output_shape_matches_input(image, segmented):
    _, markers = segmented
    assert markers.shape == image.shape[:2]

def test_segment_stores_intermediates_on_self(segmented):
    seg, _ = segmented
    for name in ["channel", "binary", "opening", "bg_mask", "dist",
                 "peaks", "marker_mask", "unknown", "markers0", "markers"]:
        assert getattr(seg, name) is not None, f"self.{name} was not stored"

def test_segment_background_marker_is_the_background_mask(segmented):
    seg, _ = segmented
    assert np.array_equal(seg.markers0 == 1, seg.bg_mask == 255)

def test_segment_splits_oranges_that_thresholding_merged(segmented):
    """Thresholding leaves the touching oranges in a few large merged regions;
    watershed must split them into many separate oranges."""
    seg, markers = segmented
    n, _, stats, _ = cv2.connectedComponentsWithStats(seg.opening)
    large_regions = int(np.sum(stats[1:, 4] > 2000))
    n_basins = len(np.unique(markers[markers > 1]))
    assert large_regions <= 5
    assert n_basins >= 25

def test_segment_basins_are_orange_sized(segmented):
    """An orange in this image is about 100 px across (~8000 px area)."""
    _, markers = segmented
    areas = np.unique(markers[markers > 1], return_counts=True)[1]
    assert 5000 <= np.median(areas) <= 15000

def test_segment_peak_spacing_affects_basin_count(image):
    close = WatershedSegmenter(peak_spacing=20).segment(image)
    wide  = WatershedSegmenter(peak_spacing=80).segment(image)
    assert len(np.unique(close[close > 1])) > len(np.unique(wide[wide > 1]))
