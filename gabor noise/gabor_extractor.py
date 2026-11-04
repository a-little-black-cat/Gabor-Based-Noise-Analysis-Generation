"""Feature extraction for creating variations of uploaded texture images."""

import csv
import os

import cv2 as cv
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
from pylette import extract_colors
source = None
gray_img = None


def _gabor_kernel_l2(frequency, theta, sigma_x, sigma_y, n_stds=3.5):
    """Complex Gabor kernel (skimage convention) normalised to unit L2 energy.

    skimage's gabor_kernel is normalised to unit envelope *area*, so its output
    level changes with sigma and small sigmas always win an energy ranking.
    Unit L2 energy makes every band comparable (white noise -> energy 1).
    """
    half_x = int(np.ceil(n_stds * max(abs(sigma_x * np.cos(theta)), abs(sigma_y * np.sin(theta)), 1)))
    half_y = int(np.ceil(n_stds * max(abs(sigma_y * np.cos(theta)), abs(sigma_x * np.sin(theta)), 1)))
    y, x = np.mgrid[-half_y:half_y + 1, -half_x:half_x + 1]
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    g = np.exp(-0.5 * (xr**2 / sigma_x**2 + yr**2 / sigma_y**2)) * np.exp(1j * 2 * np.pi * frequency * xr)
    return g / np.sqrt(np.sum(np.abs(g) ** 2))


def sigma_for_frequency(frequency, bandwidth_octaves=1.0, aspect=1.6):
    """Envelope sigmas (pixels) tied to the carrier wavelength (constant Q)."""
    k = (2.0**bandwidth_octaves + 1.0) / (2.0**bandwidth_octaves - 1.0)
    sigma_x = np.sqrt(np.log(2) / 2.0) / np.pi * k / frequency
    return float(sigma_x), float(sigma_x * aspect)


def _circular_filter(field, kernel):
    """Circular (wrap-around) convolution via FFT; output has field's shape."""
    height, width = field.shape
    padded = np.zeros((height, width), dtype=np.complex128)
    kernel_height, kernel_width = kernel.shape
    padded[:kernel_height, :kernel_width] = kernel
    padded = np.roll(padded, (-(kernel_height // 2), -(kernel_width // 2)), axis=(0, 1))
    return np.fft.ifft2(np.fft.fft2(field) * np.fft.fft2(padded))


def _rank_normalize(values):
    """Map values to (0, 1) by rank (histogram equalisation); outlier-proof."""
    flat = values.ravel()
    ranks = np.empty(flat.size, dtype=np.float64)
    ranks[np.argsort(flat, kind="stable")] = (np.arange(flat.size) + 0.5) / flat.size
    return ranks.reshape(values.shape)


def export_gabor_kernels_csv(kernels, file_path):
    """Export kernel coefficients as one row per coefficient."""
    with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("kernel", "row", "column", "coefficient"))
        for kernel_index, kernel in enumerate(kernels, start=1):
            for row, column in np.ndindex(kernel.shape):
                writer.writerow((kernel_index, row, column, float(kernel[row, column])))



def extract_gabor_parameters(image, max_components=None, n_frequencies=6, n_orientations=8):
    """Extract JSON-friendly noise and appearance parameters from an image."""
    global gray_img
    global source

    source = (
        cv.imread(os.fspath(image), cv.IMREAD_COLOR)
        if isinstance(image, (str, os.PathLike))
        else image
    )
    if source is None or source.ndim != 3:
        raise ValueError("Could not read a color texture image")

    image_float = source.astype(np.float32) / 255.0
    gray_img = cv.cvtColor(image_float, cv.COLOR_BGR2GRAY)
    gray_img -= gray_img.mean()
    gray_img /= gray_img.std() + 1e-8

    power_spectrum = get_power_spectrum()

    # Measure how much variance sits in each (frequency, orientation) band of a
    # constant-Q filter bank.  The whole bank is kept and weighted by its share,
    # instead of keeping the top-N raw energies (which always picked the lowest
    # frequency and the smallest sigma).  A Hann window removes the false
    # horizontal/vertical cross that edge discontinuities add to the spectrum.
    height, width = gray_img.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    windowed = gray_img * window
    window_power = float(np.mean(window**2))
    frequencies = np.geomspace(0.02, 0.4, n_frequencies)
    orientations = np.arange(n_orientations, dtype=np.float32) * np.pi / n_orientations
    components = []
    for frequency in frequencies:
        sigma_x, sigma_y = sigma_for_frequency(frequency)
        for theta in orientations:
            kernel = _gabor_kernel_l2(frequency, theta, sigma_x, sigma_y)
            if max(kernel.shape) > min(height, width):
                continue  # kernel larger than the image: band not measurable
            response = _circular_filter(windowed, kernel)
            energy = float(np.mean(np.abs(response) ** 2) / window_power)
            components.append({
                "frequency": float(frequency),
                "theta": float(theta),
                "sigma_x": sigma_x,
                "sigma_y": sigma_y,
                "energy": energy,
                "response_std": float(np.sqrt(energy)),
            })
    if not components:
        raise ValueError("Image is too small for the Gabor filter bank")
    total_energy = sum(item["energy"] for item in components) + 1e-12
    for item in components:
        item["share"] = item["energy"] / total_energy
    components.sort(key=lambda item: item["energy"], reverse=True)
    if max_components:
        components = components[:max_components]

    rgb = cv.cvtColor(image_float, cv.COLOR_BGR2RGB)
    luminance = cv.cvtColor(image_float, cv.COLOR_BGR2GRAY)
    highlights = np.clip(luminance - 0.8, 0.0, 0.2) / 0.2
    primary = components[0]
    palette_image = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    palette = extract_palette_from_image(palette_image, max_num_colors=8)
    power_spectrum = get_power_spectrum()
    



    return {
        "frequency": primary["frequency"],
        "theta": primary["theta"],
        "sigma_x": primary["sigma_x"],
        "sigma_y": primary["sigma_y"],
        "noise_amplitude": primary["response_std"],
        "components": components,
        "color_mean": np.mean(rgb, axis=(0, 1)).tolist(),
        "color_std": np.std(rgb, axis=(0, 1)).tolist(),
        "luminance_mean": float(np.mean(luminance)),
        "luminance_std": float(np.std(luminance)),
        "specular_mean": float(np.mean(highlights)),
        "specular_std": float(np.std(highlights)),
        "energy": float(primary["energy"]),
        "image_size": {"height": int(source.shape[0]), "width": int(source.shape[1])},
        "uv_map": None,
        "palette": [
            {
                "rgb": color.rgb,
                "hex": color.hex,
                "hsv": color.hsv,
                "frequency": float(color.frequency),
            }
            for color in palette.colors
        ],
        "power_spectrum": power_spectrum,

    }


def compute_feats(image, kernels):
    """Compute mean and variance responses for a set of Gabor kernels."""
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for index, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode="wrap")
        feats[index] = filtered.mean(), filtered.var()
    return feats

def extract_palette_from_image(image, max_num_colors=8):
    palette = extract_colors(image, max_num_colors)
    for color in palette.colors:
        print(f"RGB: {color.rgb}")
        print(f"Hex: {color.hex}")
        print(f"HSV: {color.hsv}")
        print(f"Frequency: {color.frequency:.2%}")
    palette.to_json(filename="palette.json", colorspace='hsv')
    return palette

def create_color_variation(image, color_mean=None, color_std=None, palette=None):
    """Create a color variation using extracted palette colors when available."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a color image with 3 channels.")

    # Convert to float32 for processing
    image_float = image.astype(np.float32) / 255.0

    if palette:
        palette_rgb = np.asarray(
            [color["rgb"] for color in palette], dtype=np.float32
        ) / 255.0
        palette_luminance = (
            0.2126 * palette_rgb[:, 0]
            + 0.7152 * palette_rgb[:, 1]
            + 0.0722 * palette_rgb[:, 2]
        )
        order = np.argsort(palette_luminance)
        palette_rgb = palette_rgb[order]
        palette_luminance = palette_luminance[order]

        luminance = (
            0.2126 * image_float[:, :, 0]
            + 0.7152 * image_float[:, :, 1]
            + 0.0722 * image_float[:, :, 2]
        )
        luminance_min, luminance_max = luminance.min(), luminance.max()
        if luminance_max - luminance_min < 1e-8:
            normalized_luminance = np.full_like(luminance, 0.5)
        else:
            normalized_luminance = (luminance - luminance_min) / (
                luminance_max - luminance_min
            )
        palette_positions = np.linspace(0.0, 1.0, len(palette_rgb))
        color_indices = np.abs(
            normalized_luminance[..., None] - palette_positions
        ).argmin(axis=2)
        new_image = palette_rgb[color_indices]
        return np.uint8(np.clip(new_image, 0.0, 1.0) * 255.0)

    if color_mean is None or color_std is None:
        raise ValueError("Provide either a palette or color mean and standard deviation.")
    
    # Calculate current mean and std deviation
    current_mean = np.mean(image_float, axis=(0, 1))
    current_std = np.std(image_float, axis=(0, 1))
    
    # Normalize the image
    normalized_image = (image_float - current_mean) / (current_std + 1e-8)
    
    # Apply new mean and std deviation
    new_image = normalized_image * color_std + color_mean
    
    # Clip values to [0, 1] range and convert back to uint8
    new_image_clipped = np.clip(new_image, 0.0, 1.0)
    return (new_image_clipped * 255).astype(np.uint8)


def normalize(array):
    """Normalize an array to the range [0, 1]."""
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val - min_val < 1e-8:
        return np.zeros_like(array)
    return (array - min_val) / (max_val - min_val)

def get_power_spectrum(image=None):
    """Compute the power spectrum of a grayscale image."""
    if image is None:
        if gray_img is None:
            raise ValueError("No grayscale image is available")
        gray = gray_img
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    f_transform = np.fft.fft2(gray)
    f_shifted = np.fft.fftshift(f_transform)
    power_spectrum = np.abs(f_shifted) ** 2 #Array of power spectrum values
    power_spectrum = normalize(power_spectrum)



    return power_spectrum



def generate_gabor_noise(
    height,
    width,
    gabor_parameters,
    seed=None,
    use_all_components=True,
    frequency_scale=1.0,
    sigma_scale=1.0,
    density_scale=1.0,
    impulses_per_kernel=12.0,
    envelope_size=None,
    subbands=None,
):
    """2-D sparse-convolution Gabor noise

    random impulses placed using Poisson process, with phase randomized uniformly in [0, 2pi].
    The output is a sum of Gabor functions.


    ``1 / sqrt(lambda) * sum(g(x - x_i, phi_i))``.

    ``poisson_density`` is the expected number of impulses per pixel.  When it
    is not present in the extracted parameters, the density is chosen so that
    about ``impulses_per_kernel * density_scale`` impulses fall under one
    kernel footprint (10+ gives continuous noise; ~0.1 gives isolated dots).
    ``lambda`` is also accepted as an alias for ``poisson_density``.
    Each component is standardised to unit variance and weighted by
    ``sqrt(energy share)``, so the output has unit variance overall.

    ``sigma_x`` and ``sigma_y`` are the envelope standard deviations in pixels.
    ``envelope_size`` can override them globally as a scalar or ``(x, y)`` pair.
    ``subbands`` can be a sequence of dictionaries with ``min_frequency`` and
    ``max_frequency`` bounds; each component is generated only in its matching
    band.  A subband may also provide ``envelope_size`` and ``density_scale``.
    """

    if not isinstance(height, (int, np.integer)) or height < 1:
        raise ValueError("height must be a positive integer")
    if not isinstance(width, (int, np.integer)) or width < 1:
        raise ValueError("width must be a positive integer")
    if frequency_scale <= 0.0 or sigma_scale <= 0.0 or density_scale <= 0.0:
        raise ValueError("frequency_scale, sigma_scale, and density_scale must be positive")
    if envelope_size is not None:
        if isinstance(envelope_size, (int, float, np.integer, np.floating)):
            envelope_size = (float(envelope_size), float(envelope_size))
        elif len(envelope_size) == 2:
            envelope_size = (float(envelope_size[0]), float(envelope_size[1]))
        else:
            raise ValueError("envelope_size must be a positive scalar or a pair")
        if min(envelope_size) <= 0.0:
            raise ValueError("envelope_size values must be positive")
    if subbands is not None and not subbands:
        raise ValueError("subbands must contain at least one band")

    #set envelope size


    extracted_components = gabor_parameters.get("components", [])
    if use_all_components:
        components = list(extracted_components)
    elif extracted_components:
        components = [extracted_components[0]]
    else:
        components = [gabor_parameters]
    if not components:
        components = [gabor_parameters]

    rng = np.random.default_rng(seed)
    generated = np.zeros((height, width), dtype=np.float32)
    energies = np.asarray(
        [max(float(component.get("energy", 1.0)), 0.0) for component in components],
        dtype=np.float32,
    )
    weights = energies / (energies.sum() + np.finfo(np.float32).eps)

    for component, weight in zip(components, weights):
        component_frequency = float(
            component.get("frequency", gabor_parameters.get("frequency", 0.1))
        )
        matching_subband = None
        if subbands is not None:
            for subband in subbands:
                minimum = float(subband.get("min_frequency", 0.0))
                maximum = float(subband.get("max_frequency", np.inf))
                if minimum <= component_frequency < maximum:
                    matching_subband = subband
                    break
            if matching_subband is None:
                continue
        frequency = float(
            component_frequency
        ) * frequency_scale
        if matching_subband is not None and "envelope_size" in matching_subband:
            band_envelope = matching_subband["envelope_size"]
            if isinstance(band_envelope, (int, float)):
                band_envelope = (float(band_envelope), float(band_envelope))
            if len(band_envelope) != 2 or min(band_envelope) <= 0.0:
                raise ValueError("subband envelope_size must be a positive scalar or pair")
            sigma_x, sigma_y = map(float, band_envelope)
        elif envelope_size is not None:
            sigma_x, sigma_y = envelope_size
        else:
            sigma_x = float(
                component.get("sigma_x", gabor_parameters.get("sigma_x", 1.0))
            ) * sigma_scale
            sigma_y = float(
                component.get("sigma_y", gabor_parameters.get("sigma_y", 1.0))
            ) * sigma_scale
        if frequency <= 0.0 or sigma_x <= 0.0 or sigma_y <= 0.0:
            raise ValueError("Gabor frequency and sigmas must be positive")

        explicit_density = component.get("poisson_density")
        if explicit_density is None:
            explicit_density = component.get("lambda")
        if explicit_density is None:
            explicit_density = gabor_parameters.get("poisson_density")
        component_density_scale = (
            float(matching_subband["density_scale"])
            if matching_subband is not None and "density_scale" in matching_subband
            else density_scale
        )
        density = (
            float(explicit_density)
            if explicit_density is not None
            else component_density_scale * impulses_per_kernel / (2.0 * np.pi * sigma_x * sigma_y)
        )
        if density <= 0.0:
            raise ValueError("poisson_density must be positive")

        point_count = rng.poisson(height * width * density)
        impulse_field = np.zeros((height, width), dtype=np.complex64)
        if point_count:
            rows = rng.integers(0, height, size=point_count)
            columns = rng.integers(0, width, size=point_count)
            phases = rng.uniform(0.0, 2.0 * np.pi, size=point_count)
            np.add.at(
                impulse_field,
                (rows, columns),
                np.exp(1j * phases).astype(np.complex64),
            )

        kernel = _gabor_kernel_l2(
            frequency,
            float(component.get("theta", gabor_parameters.get("theta", 0.0))),
            sigma_x,
            sigma_y,
        )
        band = np.real(_circular_filter(impulse_field, kernel))
        band /= band.std() + 1e-12
        generated += (np.sqrt(weight) * band).astype(np.float32)

    return generated

def gabor_function(x, K, a, w, phi):
    """Compute the Gabor function value at position x."""
    return K * np.exp(-np.pi * a**2 * np.abs(x)**2) * np.cos(2 * np.pi * w * x + phi)

def colorize_gabor_noise(gray_gabor_noise, parameters):
    """Colour noise with the extracted palette; always returns uint8 RGB.

    The noise is rank-normalised (outlier-proof, unlike min-max), the palette
    is sorted by luminance, and each colour covers a span of the ranks equal to
    its palette frequency, so colour proportions match the source image.
    """
    palette = parameters.get("palette", [])
    ranks = _rank_normalize(np.asarray(gray_gabor_noise, dtype=np.float64))

    if not palette:
        gray = np.uint8(np.clip(ranks * 255.0, 0, 255))
        return np.repeat(gray[:, :, np.newaxis], 3, axis=2)

    palette_rgb = np.asarray([color["rgb"] for color in palette], dtype=np.float32) / 255.0
    weights = np.asarray([float(color.get("frequency", 1.0)) for color in palette], dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)
    luminance = (
        0.2126 * palette_rgb[:, 0] + 0.7152 * palette_rgb[:, 1] + 0.0722 * palette_rgb[:, 2]
    )
    order = np.argsort(luminance)
    palette_rgb, weights = palette_rgb[order], weights[order]
    positions = np.cumsum(weights) - weights / 2.0  # centre of each colour's rank span

    output = np.empty((*ranks.shape, 3), dtype=np.float32)
    for channel in range(3):
        output[..., channel] = np.interp(ranks, positions, palette_rgb[:, channel])
    return np.uint8(np.clip(output * 255.0, 0, 255))

def estimate_dominant_frequencies(
    power_spectrum,
    num_frequencies=5,
    bins=64,
    min_frequency=0.01,
):
    """Estimate dominant radial frequencies from ``get_power_spectrum`` output."""
    power_spectrum = np.asarray(power_spectrum, dtype=np.float32)
    if power_spectrum.ndim != 2:
        raise ValueError("power_spectrum must be a two-dimensional array")
    if num_frequencies < 1:
        raise ValueError("num_frequencies must be at least 1")
    if bins < 1:
        raise ValueError("bins must be at least 1")
    if not 0.0 <= min_frequency < 0.5:
        raise ValueError("min_frequency must be in the range [0, 0.5)")

    height, width = power_spectrum.shape
    y, x = np.indices(power_spectrum.shape)
    x_frequency = (x - width // 2) / width
    y_frequency = (y - height // 2) / height
    radial_frequency = np.sqrt(x_frequency**2 + y_frequency**2)

    frequency_edges = np.linspace(min_frequency, 0.5, bins + 1)
    candidates = []
    for lower, upper in zip(frequency_edges[:-1], frequency_edges[1:]):
        mask = (
            (radial_frequency >= lower)
            & (radial_frequency < upper)
        )
        if np.any(mask):
            candidates.append({
                "frequency": float((lower + upper) / 2.0),
                "power": float(np.mean(power_spectrum[mask])),
            })

    candidates.sort(key=lambda item: item["power"], reverse=True)
    return candidates[:num_frequencies]