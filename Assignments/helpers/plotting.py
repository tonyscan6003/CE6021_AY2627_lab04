# -*- coding: utf-8 -*-
"""Shared Plotly reporting helpers for CE6021 lab evaluation scripts."""
import base64
import os

import cv2
import numpy as np
import plotly.io as pio


def png_source(img):
    """Encode an image as a PNG data URI, for go.Image(source=...).

    Embedding images this way keeps the HTML report small: go.Image(z=img)
    stores every pixel uncompressed, which makes multi-figure reports tens of
    megabytes and can stop them loading fully in the browser.

    Args:
        img (np.ndarray): H x W greyscale or H x W x 3 RGB image (any numeric
            dtype; values are clipped to 0-255).

    Returns:
        str: "data:image/png;base64,..." string.
    """
    img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Could not encode image as PNG.")
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")


def jpeg_source(img, quality=90):
    """Encode an image as a JPEG data URI, for go.Image(source=...).

    Use this for photographs (and overlays drawn on photographs): a photo is
    typically 10x smaller as JPEG than as PNG. Keep png_source for binary,
    edge and label images — they compress well as PNG and JPEG would blur
    their sharp edges.

    Args:
        img (np.ndarray): H x W greyscale or H x W x 3 RGB image (any numeric
            dtype; values are clipped to 0-255).
        quality (int): JPEG quality, 0-100.

    Returns:
        str: "data:image/jpeg;base64,..." string.
    """
    img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Could not encode image as JPEG.")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def save_html(*figs, output_path):
    """Write one or more Plotly figures to a single combined HTML file.

    Args:
        *figs: One or more plotly.graph_objects.Figure instances.
        output_path (str): Path to write the combined HTML report to.
    """
    parts = [
        pio.to_html(fig, full_html=False,
                    include_plotlyjs=("cdn" if i == 0 else False))
        for i, fig in enumerate(figs)
    ]
    title = os.path.basename(output_path)
    html = (
        "<!DOCTYPE html><html>"
        "<head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>body{{font-family:sans-serif;padding:20px}}"
        "hr{{margin:30px 0;border:1px solid #ccc}}</style></head>"
        "<body>" + "<hr>".join(parts) + "</body></html>"
    )
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"\nResults saved → {output_path}")
    print("Open in VS Code: right-click → Open with Live Server")
    print("         or run: python -m http.server 5000")
