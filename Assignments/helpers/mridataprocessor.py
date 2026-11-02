# -*- coding: utf-8 -*-
"""MRIDataProcessor — utility for creating/saving corrupted MRI k-space test data.

This module is an instructor/helper tool, not student-facing.  It can be run
directly to regenerate Data/corrupted_mrimage1d.txt from Data/mrimage1d.txt:

    python Assignments/helpers/mridataprocessor.py
"""

import math
import os
import sys
import numpy as np

# Allow running as a top-level script as well as being imported as a package module
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, os.path.dirname(_here))  # adds Assignments/ to path

from helpers.dataloader import get_data_path


class MRIDataProcessor:
    """Generates corrupted MRI k-space data for lab exercises.

    Workflow:  clean k-space  →  IFFT  →  add sinusoidal corruption
               →  FFT  →  save corrupted k-space to text file.
    """

    def __init__(self,
                 input_filepath=None,
                 output_filepath=None,
                 shape=(256, 256)):
        self.input_filepath  = input_filepath  or get_data_path("mrimage1d.txt")
        self.output_filepath = output_filepath or get_data_path("corrupted_mrimage1d.txt")
        self.shape = shape

    # ------------------------------------------------------------------
    # Static helpers (usable without an instance)
    # ------------------------------------------------------------------

    @staticmethod
    def load_kspace_1d(filepath):
        """Load a two-column (Real, Imag) text file and return a 1D complex array.

        Args:
            filepath (str): Path to the text file.

        Returns:
            np.ndarray: 1D complex array (length = number of rows in file).
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        data = np.loadtxt(filepath)
        if data.ndim == 2 and data.shape[1] >= 2:
            return data[:, 0] + 1j * data[:, 1]
        return data.astype(complex)

    # ------------------------------------------------------------------
    # Instance methods
    # ------------------------------------------------------------------

    def load_data(self):
        """Load clean k-space from input_filepath and reshape to self.shape."""
        return MRIDataProcessor.load_kspace_1d(self.input_filepath).reshape(self.shape)

    def corrupt_image(self, img):
        """Add a low-amplitude sinusoidal pattern to corrupt the image.

        Args:
            img (np.ndarray): Grayscale image (values in [0, 255]).

        Returns:
            np.ndarray: Corrupted image normalised to [0, 255].
        """
        M, N = img.shape[1], img.shape[0]
        X, Y = np.meshgrid(np.arange(M), np.arange(N))
        k, l = 19, 20
        fxy = np.exp(2j * math.pi * (k * X / M + l * Y / N))
        op_img = img + 100 * np.real(fxy)
        return 255 * op_img / np.max(op_img)

    def save_data(self, k_space_data):
        """Flatten 256×256 complex data and save as a 1D two-column text file."""
        complex_1d = k_space_data.flatten()
        output_data = np.column_stack((np.real(complex_1d), np.imag(complex_1d)))
        os.makedirs(os.path.dirname(self.output_filepath), exist_ok=True)
        np.savetxt(self.output_filepath, output_data, fmt='%.8e', delimiter=' ')

    def process(self):
        """Orchestrate: load → IFFT → corrupt → FFT → save."""
        print(f"1. Loading MRI data from {self.input_filepath}...")
        k_space_data = self.load_data()

        print("2. Applying IFFT to convert to image domain...")
        image_complex = np.fft.ifft2(np.fft.ifftshift(k_space_data))
        image_domain = np.abs(image_complex)
        image_domain = 255 * image_domain / np.max(image_domain)

        print("3. Corrupting image with sinusoidal pattern...")
        corrupted_image = self.corrupt_image(image_domain)

        print("4. Applying FFT to convert back to k-space...")
        corrupted_k_space = np.fft.fftshift(np.fft.fft2(corrupted_image))

        print(f"5. Saving corrupted data to {self.output_filepath}...")
        self.save_data(corrupted_k_space)
        print("Done.")


if __name__ == "__main__":
    processor = MRIDataProcessor()
    try:
        processor.process()
    except Exception as e:
        print(f"Error: {e}")
