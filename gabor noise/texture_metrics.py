"""Metrics for comparing local colour and Gabor texture statistics."""

import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gabor


def _features(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    if image.max() > 1.0:
        image = image / 255.0
    image = np.clip(image, 0.0, 1.0)
    gray = rgb2gray(image)
    features = [
        *np.mean(image, axis=(0, 1)),
        *np.std(image, axis=(0, 1)),
        float(gray.mean()),
        float(gray.std()),
    ]
    for frequency in (0.05, 0.1, 0.2, 0.35):
        for theta in (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
            real, imaginary = gabor(gray, frequency=frequency, theta=theta)
            magnitude = np.hypot(real, imaginary)
            features.extend((float(magnitude.mean()), float(magnitude.std())))
    return np.asarray(features, dtype=np.float32)


def gabor_texture_distance(reference, generated):
    """Return a normalized distance between two images' colour/Gabor statistics."""
    reference_features = _features(reference)
    generated_features = _features(generated)
    scale = np.maximum(np.abs(reference_features), 1e-3)
    return float(np.mean(np.abs(reference_features - generated_features) / scale))
