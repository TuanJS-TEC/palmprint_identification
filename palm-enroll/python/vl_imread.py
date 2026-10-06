"""NumPy re-implementation of vl_imreadjpeg(..., 'Resize', [H W]) (MatConvNet beta25).

This is the preprocessing used to train and evaluate EE-PRnet
(EE-PRnet/getBatchNclassSpecial.m, functions/buildFeat.m):

  1. crop: with the default CropSize [1 1] / CropAnisotropy [1 1] /
     CropLocation 'center', the largest centred crop with the aspect ratio of
     the output is taken (vl_imreadjpeg.cu): a non-square image is NOT stretched.
  2. resize: separable triangle filter, vertical pass then horizontal pass, in
     float32 without rounding (bits/impl/imread_helpers.hpp, ImageResizeFilter):
       alpha = crop / out, u = alpha * v + 0.5 * (alpha - 1) + cropOffset
       support = 2 * max(alpha, 1), taps k = ceil(u - support / 2) + r
       h = max(0, 1 - |u - k| / max(alpha, 1)), weights normalised to sum 1
     taps outside the image are folded onto the border pixel (replicate).

Pillow's BILINEAR uses the same centres and kernel but drops out-of-image taps
instead of replicating them, so it differs on the first/last rows and columns.
palm-enroll/src/preprocess.cpp implements exactly the same algorithm.
"""
import numpy as np


def crop_geometry(in_h, in_w, out_h, out_w):
    """(x, y, w, h) of the centred isotropic crop (vl_imreadjpeg.cu)."""
    scale = min(in_w / out_w, in_h / out_h)
    cw = int(min(np.floor(out_w * scale + 0.5), in_w))  # C round(): half away from zero
    ch = int(min(np.floor(out_h * scale + 0.5), in_h))
    return (in_w - cw) // 2, (in_h - ch) // 2, cw, ch


def resize_matrix(out_size, in_size, crop_size, crop_offset):
    """Dense [out_size, in_size] matrix of ImageResizeFilter."""
    f32 = np.float32
    alpha = f32(crop_size) / f32(out_size)
    beta = f32(0.5) * (alpha - f32(1)) + f32(crop_offset)
    support, taps = f32(2), 2
    if alpha > 1:
        support = support * alpha
        taps = int(np.ceil(support))
    m = np.zeros((out_size, in_size), np.float32)
    for v in range(out_size):
        u = alpha * f32(v) + beta
        start = int(np.ceil(u - support / f32(2)))
        mass = f32(0)
        for r in range(taps):
            k = start + r
            delta = u - f32(k)
            if alpha > 1:
                delta = delta / alpha
            h = max(f32(0), f32(1) - abs(delta))
            m[v, min(max(k, 0), in_size - 1)] += h
            mass += h
        m[v] /= mass
    return m


def vl_resize(img, out_h, out_w):
    """img: float32 H x W x C (0..255). Returns float32 out_h x out_w x C."""
    img = np.asarray(img, np.float32)
    x, y, cw, ch = crop_geometry(img.shape[0], img.shape[1], out_h, out_w)
    mv = resize_matrix(out_h, img.shape[0], ch, y)
    mh = resize_matrix(out_w, img.shape[1], cw, x)
    tmp = np.einsum("vh,hwc->vwc", mv, img)
    return np.einsum("uw,vwc->vuc", mh, tmp).astype(np.float32)
