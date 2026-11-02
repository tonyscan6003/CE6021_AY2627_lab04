# -*- coding: utf-8 -*-
"""Evaluate Lab 04 — Watershed Segmentation.

Run from the repo root:
    python Assignments/lab04/evaluate_lab04.py

Runs each step of WatershedSegmenter in turn and adds a figure for every step
that works. If a step isn't implemented yet, it stops there and still saves
the figures made so far — so you can run this after finishing each step.
Results are written to a single HTML file next to this script.
"""
import os
import sys
import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB_DIR)
sys.path.insert(0, os.path.join(LAB_DIR, ".."))

from lab04 import WatershedSegmenter
from helpers.dataloader import get_data_path, load_image
from helpers.plotting import save_html, png_source, jpeg_source

HOVER = "x: %{x}<br>y: %{y}<extra></extra>"
RED, CYAN = (255, 0, 0), (0, 255, 255)

def encode(img, photo):
    """Photographs as JPEG (much smaller); binary and label images as PNG (sharp edges)."""
    return jpeg_source(img) if photo else png_source(img)

def plot_image(img, title, photo=False):
    """A single-panel image figure."""
    fig = go.Figure(go.Image(source=encode(img, photo), hovertemplate=HOVER))
    fig.update_layout(title_text=title, height=650)
    return fig

def plot_pair(left, right, left_title, right_title, title, left_photo=True, right_photo=True):
    """Two images side by side."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=[left_title, right_title])
    fig.add_trace(go.Image(source=encode(left, left_photo), hovertemplate=HOVER), row=1, col=1)
    fig.add_trace(go.Image(source=encode(right, right_photo), hovertemplate=HOVER), row=1, col=2)
    fig.update_layout(title_text=title, height=650)
    return fig

def plot_distance(dist, title):
    """The distance map as a heatmap — hover to read the distance (in pixels).
    Shown at half size, rounded to whole pixels, to keep the report small."""
    small = cv2.resize(dist, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    small = np.round(small).astype(np.uint8)
    fig = go.Figure(go.Heatmap(
        z=small, x=np.arange(small.shape[1]) * 2, y=np.arange(small.shape[0]) * 2,
        colorscale="Viridis", colorbar=dict(title="px"),
        hovertemplate="x: %{x}<br>y: %{y}<br>distance: %{z} px<extra></extra>"))
    fig.update_yaxes(autorange="reversed", scaleanchor="x")
    fig.update_layout(title_text=title, height=650)
    return fig

def colour_labels(markers):
    """One random colour per label; background dark grey, boundaries white."""
    rng = np.random.default_rng(0)
    palette = rng.integers(60, 256, size=(markers.max() + 2, 3), dtype=np.uint8)
    out = palette[np.clip(markers, 0, None)]
    out[markers == 1]  = (40, 40, 40)
    out[markers == 0]  = (0, 0, 0)
    out[markers == -1] = (255, 255, 255)
    return out

def draw_points(image, points, colour):
    out = image.copy()
    for row, col in points:
        cv2.circle(out, (int(col), int(row)), 6, colour, -1)
    return out

def draw_boundaries(image, markers):
    out = image.copy()
    thick = cv2.dilate((markers == -1).astype(np.uint8), np.ones((3, 3), np.uint8))
    out[thick > 0] = RED
    return out

def evaluate():
    image = load_image(get_data_path("Oranges_in_a_box_crop.jpg"))

    seg = WatershedSegmenter()
    figs = [plot_image(image, "Input image — a box of oranges", photo=True)]
    step = ""

    try:
        step = "Step 1 (colour_channel)"
        channel = seg.colour_channel(image)
        figs.append(plot_image(channel, "Step 1: Red minus blue — oranges bright, newspaper and gaps dark", photo=True))
        print(f"{step} done.")

        step = "Step 2 (threshold)"
        binary = seg.threshold(channel)
        figs.append(plot_image(binary, "Step 2: Otsu threshold — the touching oranges merge together"))
        print(f"{step} done.")

        step = "Step 3 (remove_noise)"
        opening = seg.remove_noise(binary)
        _, _, stats, _ = cv2.connectedComponentsWithStats(opening)
        n_regions = int(np.sum(stats[1:, 4] > 2000))
        figs.append(plot_image(opening, f"Step 3: After opening — speckle removed; only "
                                         f"{n_regions} large white regions for ~30 oranges"))
        print(f"{step} done — {n_regions} large connected regions (thresholding alone can't count the oranges).")

        step = "Step 4 (background_mask)"
        bg_mask = seg.background_mask(opening)
        figs.append(plot_image(bg_mask, f"Step 4: Background mask — white is certainly not an orange, "
                                        f"kept {seg.dilate_iterations} px away from every orange"))
        print(f"{step} done.")

        step = "Step 5 (distance_transform)"
        dist = seg.distance_transform(opening)
        figs.append(plot_distance(dist, "Step 5: Distance transform — one bright 'hill' per orange (hover to read distances)"))
        print(f"{step} done — largest distance {dist.max():.0f} px.")

        step = "Step 6 (orange_markers)"
        marker_mask = seg.orange_markers(dist)
        figs.append(plot_image(draw_points(image, seg.peaks, CYAN),
                               f"Step 6: {len(seg.peaks)} markers — one at the local maximum of each orange",
                               photo=True))
        print(f"{step} done — {len(seg.peaks)} markers.")

        step = "Step 7 (unknown_region)"
        unknown = seg.unknown_region(bg_mask, marker_mask)
        figs.append(plot_image(unknown, "Step 7: Unknown region — white is what watershed has to decide"))
        print(f"{step} done.")

        step = "Step 8 (create_markers)"
        markers0 = seg.create_markers(marker_mask, unknown)
        figs.append(plot_image(colour_labels(markers0),
                               "Step 8: Initial markers — one colour per orange marker, grey = background marker, black = unknown"))
        print(f"{step} done — {int(markers0.max()) - 1} markers.")

        step = "Step 9 (apply_watershed)"
        markers = seg.apply_watershed(image, markers0)
        figs.append(plot_image(draw_boundaries(image, markers),
                               "Step 9: Watershed boundaries (red) — the floods meet at the seams between oranges", photo=True))
        print(f"{step} done.")

        step = "Step 10 (segment)"
        markers = seg.segment(image)
        n_oranges = len(np.unique(markers[markers > 1]))
        figs.append(plot_pair(
            colour_labels(markers), draw_boundaries(image, markers),
            "One colour per detected orange", "Boundaries on the image",
            f"Step 10: Full pipeline — {n_oranges} oranges (thresholding alone gave {n_regions} regions)",
            left_photo=False))
        print(f"{step} done — {n_oranges} oranges detected.")

    except NotImplementedError:
        print(f"\n{step} is not implemented yet — stopping here.")
        print("Figures for the earlier steps are still saved below.")

    save_html(*figs, output_path=os.path.join(LAB_DIR, "lab04_results.html"))

if __name__ == "__main__":
    evaluate()
