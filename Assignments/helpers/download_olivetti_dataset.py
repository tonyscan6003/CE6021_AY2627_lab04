#!/usr/bin/env python3
"""download_olivetti_dataset.py — Load and prepare the Olivetti Faces dataset.

The dataset is fetched from sklearn (cached to Data/) and split into
train / test sets.  The data is transposed to the form expected by
EigenfaceRecognizer:  n_features × n_samples.

Usage:
    from helpers.download_olivetti_dataset import OlivettiDataset
    dataset = OlivettiDataset()
    X_train, X_test, y_train, y_test = dataset.get_data()
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataloader import DATA_DIR  # noqa: F401

from sklearn import datasets
from sklearn.model_selection import train_test_split
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class OlivettiDataset:
    """Load the Olivetti Faces dataset and provide train/test splits.

    The Olivetti dataset contains 400 greyscale 64×64 face images of 40 people
    (10 images per person).  Pixel values are float32 in [0, 1].

    Data arrays are transposed to shape (n_features, n_samples) to match the
    column-vector convention used in eigenface calculations.

    Args:
        test_size    (float): Fraction of data held out for testing (default 0.25).
        random_state (int):   Random seed for reproducibility (default 0).
    """

    def __init__(self, test_size=0.25, random_state=0):
        self.test_size    = test_size
        self.random_state = random_state
        self.faces        = datasets.fetch_olivetti_faces(data_home=DATA_DIR)
        self.image_shape  = (64, 64)
        self._prepare_data()

    def _prepare_data(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.faces.data, self.faces.target,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        # Transpose to (n_features, n_samples) for eigenface maths
        self.X_train = X_train.T
        self.X_test  = X_test.T
        self.y_train = y_train
        self.y_test  = y_test
        side = int(np.sqrt(self.X_train.shape[0]))
        self.image_shape = (side, side)

    def get_data(self):
        """Return (X_train, X_test, y_train, y_test).

        X arrays are shape (4096, n_samples); y arrays are shape (n_samples,).
        """
        return self.X_train, self.X_test, self.y_train, self.y_test

    def plot_samples(self, n_rows=2, n_cols=5):
        """Return a plotly figure showing sample face images with identity labels.

        Args:
            n_rows (int): Number of rows of faces to show (default 2).
            n_cols (int): Number of columns of faces to show (default 5).

        Returns:
            go.Figure: Plotly figure.
        """
        n      = n_rows * n_cols
        titles = [f"ID: {self.faces.target[i]}" for i in range(n)]
        fig    = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles)
        for idx in range(n):
            r, c   = divmod(idx, n_cols)
            img8   = (self.faces.images[idx] * 255).astype(np.uint8)
            rgb    = np.stack([img8, img8, img8], axis=-1)
            fig.add_trace(go.Image(z=rgb), row=r + 1, col=c + 1)
        fig.update_layout(title_text="Olivetti Face Samples", height=300)
        return fig
