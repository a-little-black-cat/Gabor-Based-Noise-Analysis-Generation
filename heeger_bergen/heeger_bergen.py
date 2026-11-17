"""
Heeger-Bergen (1995) pyramid-based texture synthesis.

Pieces:
  1. A steerable pyramid built in the Fourier domain (Simoncelli & Freeman).
     Undecimated (every band is full size) to keep the code short.
  2. Histogram matching (sort-based, works when sizes differ).
  3. The iterative loop: match subband histograms -> collapse -> match pixels.

Dependencies: numpy (pillow + matplotlib only for the demo).
"""
import numpy as np
from math import factorial, pi


def comb(n, k):
    """Return an integer binomial coefficient on older Python versions."""
    if k < 0 or k > n:
        return 0
    return factorial(n) // (factorial(k) * factorial(n - k))


# ----------------------------------------------------------------------
# 1. Steerable pyramid filters
# ----------------------------------------------------------------------
def build_filters(shape, n_scales=4, n_orient=4):
    """
    Returns (hi0, bands, lo):
      hi0   : highpass residual filter           (H, W)
      bands : bands[s][k] oriented filters       (H, W) each
      lo    : final lowpass residual filter      (H, W)
    Radial parts satisfy h^2 + l^2 = 1 so the whole thing is a tight frame.
    """
    H, W = shape
    wy = 2 * pi * np.fft.fftfreq(H)[:, None]
    wx = 2 * pi * np.fft.fftfreq(W)[None, :]
    r = np.sqrt(wx**2 + wy**2)
    theta = np.arctan2(wy, wx)
    r_safe = np.where(r == 0, 1e-12, r)

    def radial(s):
        # transition band between pi*2^-(s+1) and pi*2^-s (log-frequency)
        u = np.clip(np.log2(r_safe / pi) + s + 1, 0.0, 1.0)
        return np.sin(0.5 * pi * u), np.cos(0.5 * pi * u)  # highpass, lowpass

    # Angular filters: cos^(K-1), one-sided (half plane), so bands are complex
    # (real part = even-phase filter, imag part = odd-phase filter).
    K = n_orient
    m = K - 1
    # sum_k cos^(2m)(theta - pi k/K) = K * C(2m,m) / 4^m  (constant)
    # normalise so G(t)^2 + G(t+pi)^2 summed over k equals 2 (the Re() halves it)
    norm = np.sqrt(2.0 / (K * comb(2 * m, m) / 4**m))
    angular = []
    for k in range(K):
        d = (theta - pi * k / K + pi) % (2 * pi) - pi
        g = np.where(np.abs(d) < pi / 2, np.cos(d) ** m, 0.0) * norm
        angular.append(g)

    h0, cum = radial(0)          # s = 0: split off highpass residual
    hi0 = h0
    cum = cum.copy()
    bands = []
    for s in range(1, n_scales + 1):
        h, l = radial(s)
        radial_band = cum * h
        bands.append([radial_band * g for g in angular])
        cum = cum * l
    lo = cum
    return hi0, bands, lo


# ----------------------------------------------------------------------
# Analysis / synthesis
# ----------------------------------------------------------------------
def analyze(img, filters):
    hi0, bands, lo = filters
    X = np.fft.fft2(img)
    hi = np.fft.ifft2(X * hi0).real
    lo_img = np.fft.ifft2(X * lo).real
    bnd = [[np.fft.ifft2(X * B) for B in level] for level in bands]  # complex
    return hi, bnd, lo_img


def synthesize(hi, bnd, lo_img, filters):
    hi0, bands, lo = filters
    Y = np.fft.fft2(hi) * hi0 + np.fft.fft2(lo_img) * lo
    for level, flevel in zip(bnd, bands):
        for b, B in zip(level, flevel):
            Y += np.fft.fft2(b) * B
    return np.fft.ifft2(Y).real  # Re() symmetrises the one-sided bands


# ----------------------------------------------------------------------
# 2. Histogram matching
# ----------------------------------------------------------------------
def match_hist(src, ref):
    """Remap src values so its histogram equals ref's (any sizes)."""
    s = src.ravel()
    ref_sorted = np.sort(ref.ravel())
    if ref_sorted.size != s.size:  # resample the reference quantiles
        q = np.linspace(0, 1, s.size)
        ref_sorted = np.interp(q, np.linspace(0, 1, ref_sorted.size), ref_sorted)
    out = np.empty_like(s, dtype=float)
    out[np.argsort(s, kind="stable")] = ref_sorted
    return out.reshape(src.shape)


def match_complex(src, ref):
    return match_hist(src.real, ref.real) + 1j * match_hist(src.imag, ref.imag)


# ----------------------------------------------------------------------
# 3. Heeger-Bergen loop
# ----------------------------------------------------------------------
def heeger_bergen(texture, out_shape=None, n_scales=4, n_orient=4,
                  n_iter=5, seed=0):
    """texture: 2-D float array. Returns synthesized texture of out_shape."""
    rng = np.random.default_rng(seed)
    texture = texture.astype(float)
    out_shape = out_shape or texture.shape

    f_tex = build_filters(texture.shape, n_scales, n_orient)
    f_out = build_filters(out_shape, n_scales, n_orient)

    # Target statistics: the texture's pyramid (histograms of every band)
    t_hi, t_bnd, t_lo = analyze(texture, f_tex)

    # Start from white noise with the texture's pixel histogram
    x = match_hist(rng.standard_normal(out_shape), texture)

    for it in range(n_iter):
        hi, bnd, lo = analyze(x, f_out)
        hi = match_hist(hi, t_hi)
        lo = match_hist(lo, t_lo)
        bnd = [[match_complex(b, tb) for b, tb in zip(lv, tlv)]
               for lv, tlv in zip(bnd, t_bnd)]
        x = synthesize(hi, bnd, lo, f_out)
        x = match_hist(x, texture)          # re-impose pixel histogram
    return x


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from PIL import Image

    # sanity check: analysis -> synthesis must be (near) perfect
    test = np.random.default_rng(1).standard_normal((128, 128))
    f = build_filters(test.shape)
    err = np.abs(synthesize(*analyze(test, f), f) - test).max()
    print(f"reconstruction max error: {err:.2e}")

    if len(sys.argv) > 1:
        tex = np.asarray(Image.open(sys.argv[1]).convert("L"), float) / 255
        out = heeger_bergen(tex, out_shape=tex.shape, n_iter=5)
        out = np.clip((out - out.min()) / (np.ptp(out) + 1e-12), 0, 1)
        Image.fromarray((out * 255).astype(np.uint8)).save("synth.png")
        print("saved synth.png")