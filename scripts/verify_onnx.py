"""Check the exported ONNX models against an independent NumPy re-implementation
of the MatConvNet forward pass, then sanity-check recognition on NTU-PI-demo.

The NumPy reference works directly on the .mat data in MatConvNet layout
(H x W x C, filters [h,w,cin,cout]) and evaluates the TPS grid with the
original formula from TpsGridGenerator.m, so it shares no code with the
PyTorch graph used for export.

Usage:
  venv/bin/python scripts/verify_onnx.py
"""
import glob
import os
import re

import numpy as np
import onnxruntime as ort
import scipy.io as sio
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "Palmprint-Recognition-in-the-Wild-master/databases/NTU-PI-demo/flip/img")
AVG = np.array([123.68, 116.78, 103.94], np.float32)


# ---------------------------------------------------------------- NumPy reference
def ref_conv(x, f, b, pad, stride):
    f = f.reshape(f.shape + (1,) * (4 - f.ndim)) if f.ndim < 4 else f
    t, bo, l, r = np.broadcast_to(pad, 4)
    x = np.pad(x, ((t, bo), (l, r), (0, 0)))
    kh, kw, _, _ = f.shape
    sh, sw = np.broadcast_to(stride, 2)
    H = (x.shape[0] - kh) // sh + 1
    W = (x.shape[1] - kw) // sw + 1
    y = np.zeros((H, W, f.shape[3]), np.float64)
    for i in range(kh):
        for j in range(kw):
            y += np.tensordot(x[i : i + sh * H : sh, j : j + sw * W : sw], f[i, j], axes=([2], [0]))
    return (y + b.reshape(-1)).astype(np.float32)


def ref_pool(x, size, stride, pad):
    t, b, l, r = np.broadcast_to(pad, 4)
    x = np.pad(x, ((t, b), (l, r), (0, 0)), constant_values=-np.inf)
    ph, pw = np.broadcast_to(size, 2)
    sh, sw = np.broadcast_to(stride, 2)
    H = (x.shape[0] - ph) // sh + 1
    W = (x.shape[1] - pw) // sw + 1
    out = np.full((H, W, x.shape[2]), -np.inf, np.float32)
    for i in range(ph):
        for j in range(pw):
            out = np.maximum(out, x[i : i + sh * H : sh, j : j + sw * W : sw])
    return out


def ref_lrn(x, param):
    n, kappa, alpha, beta = param
    D = x.shape[2]
    lo, hi = int(np.floor((n - 1) / 2)), int(np.ceil((n - 1) / 2))
    out = np.empty_like(x)
    for k in range(D):
        s = (x[:, :, max(0, k - lo) : min(D, k + hi + 1)] ** 2).sum(2)
        out[:, :, k] = x[:, :, k] / (kappa + alpha * s) ** beta
    return out


def U(r):  # TpsGridGenerator.U
    r = r.copy()
    r[r == 0] = 1
    return r ** 2 * np.log(r ** 2)


def ref_tps(theta, blk):  # TpsGridGenerator.forward for one sample
    Xp, Yp = theta[: blk.Nkp], theta[blk.Nkp :]
    out = []
    for c in (Xp, Yp):
        w, a = blk.Li_w @ c, blk.Li_a @ c
        d = np.sqrt((blk.XG[..., None] - blk.X) ** 2 + (blk.YG[..., None] - blk.Y) ** 2)
        out.append(a[0] + a[1] * blk.XG + a[2] * blk.YG + (U(d) * w).sum(-1))
    XGp, YGp = out
    return np.stack([YGp, XGp])  # grid(1)=vertical, grid(2)=horizontal


def ref_sampler(img, grid):  # vl_nnbilinearsampler, zero outside
    H, W, _ = img.shape
    v = (grid[0] + 1) / 2 * (H - 1)
    u = (grid[1] + 1) / 2 * (W - 1)
    v0, u0 = np.floor(v).astype(int), np.floor(u).astype(int)
    out = np.zeros(grid.shape[1:] + (img.shape[2],), np.float64)
    for dv in (0, 1):
        for du in (0, 1):
            vi, ui = v0 + dv, u0 + du
            wgt = (1 - np.abs(v - vi)) * (1 - np.abs(u - ui))
            ok = (vi >= 0) & (vi < H) & (ui >= 0) & (ui < W)
            out += (wgt * ok)[..., None] * img[np.clip(vi, 0, H - 1), np.clip(ui, 0, W - 1)]
    return out.astype(np.float32)


def ref_forward(net, feeds, extra_tps=None):
    P = {p.name: p.value.astype(np.float32) for p in net.params}
    v = dict(feeds)
    for L in np.atleast_1d(net.layers):
        t, b = L.type, L.block
        ins = [L.inputs] if isinstance(L.inputs, str) else list(L.inputs)
        out = L.outputs if isinstance(L.outputs, str) else L.outputs[0]
        if t == "dagnn.Conv":
            f = P[L.params[0]].reshape(tuple(b.size))
            v[out] = ref_conv(v[ins[0]], f, P[L.params[1]], b.pad, b.stride)
        elif t == "dagnn.ReLU":
            x = v[ins[0]]
            v[out] = np.where(x > 0, x, x * b.leak)
        elif t == "dagnn.Pooling":
            v[out] = ref_pool(v[ins[0]], b.poolSize, b.stride, b.pad)
        elif t == "dagnn.LRN":
            v[out] = ref_lrn(v[ins[0]], b.param)
        elif t == "dagnn.DropOut":
            v[out] = v[ins[0]]
        elif t == "dagnnExtra.TpsGridGenerator":
            v[out] = ref_tps(v[ins[0]].reshape(-1), b)
        elif t == "dagnn.BilinearSampler":
            v[out] = ref_sampler(v[ins[0]], v[ins[1]])
    if extra_tps is not None:  # ROI-LAnet/demo3.m
        v["NNInput"] = ref_sampler(v["Input224"], ref_tps(v["theta"].reshape(-1), extra_tps))
    return v


# ---------------------------------------------------------------- helpers
def load_rgb(path, size=None):
    im = Image.open(path).convert("RGB")
    if size is not None:
        im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, np.float32)


def nchw(x):
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def class_id(path):
    return int(re.search(r"Class(\d+)", os.path.basename(path)).group(1))


def matlab_tps_block(size):
    """TpsGridGenerator('Ho',size,'Wo',size,'k',3,'lambda',0) done the matlab way."""
    class B:
        pass

    b = B()
    X, Y = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
    X, Y = X.flatten("F"), Y.flatten("F")
    K = U(np.sqrt((X[:, None] - X[None]) ** 2 + (Y[:, None] - Y[None]) ** 2))
    P = np.c_[np.ones(9), X, Y]
    Li = np.linalg.inv(np.block([[K, P], [P.T, np.zeros((3, 3))]]))
    b.X, b.Y, b.Nkp = X, Y, 9
    b.Li_w, b.Li_a = Li[:9, :9], Li[9:, :9]
    b.XG, b.YG = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    return b


def report(name, ref, got):
    err = np.abs(ref - got).max()
    rel = err / (np.abs(ref).max() + 1e-12)
    print(f"    {name:8s} max|diff| = {err:.3e}  (rel {rel:.1e})")
    return rel


# ---------------------------------------------------------------- main
def main():
    so = ort.SessionOptions()
    so.log_severity_level = 3
    files = sorted(glob.glob(os.path.join(DEMO, "test", "*")))
    sample = [files[0], files[len(files) // 2], files[-1]]
    worst = 0.0

    # 1) AlignNet vs reference
    net = sio.loadmat(os.path.join(ROOT, "models/AlignNet-epoch-25.mat"), struct_as_record=False, squeeze_me=True)["net"]
    sess = ort.InferenceSession(os.path.join(ROOT, "models/AlignNet-epoch-25.onnx"), so)
    tps112 = matlab_tps_block(112)
    print("AlignNet-epoch-25.onnx vs NumPy/MatConvNet reference")
    for f in sample:
        x56, full = load_rgb(f, 56), load_rgb(f)
        ref = ref_forward(net, {"ANInput": x56 - AVG, "Input224": full - AVG}, tps112)
        theta, roi = sess.run(None, {"input_56": nchw(x56), "input_full": nchw(full)})
        print("  ", os.path.basename(f))
        worst = max(worst, report("theta", ref["theta"].reshape(-1), theta[0]))
        worst = max(worst, report("roi", ref["NNInput"] + AVG, roi[0].transpose(1, 2, 0)))

    # 2) EE_PRnet vs reference
    net = sio.loadmat(os.path.join(ROOT, "models/EE_PRnet-epoch-60.mat"), struct_as_record=False, squeeze_me=True)["net"]
    sess = ort.InferenceSession(os.path.join(ROOT, "models/EE_PRnet-epoch-60.onnx"), so)
    print("EE_PRnet-epoch-60.onnx vs NumPy/MatConvNet reference")
    for f in sample:
        x56, x224 = load_rgb(f, 56), load_rgb(f, 224)
        ref = ref_forward(net, {"ANInput": x56 - AVG, "Input224": x224 - AVG})
        feat, logits, theta, roi = sess.run(None, {"input_56": nchw(x56), "input_224": nchw(x224)})
        print("  ", os.path.basename(f))
        worst = max(worst, report("theta", ref["theta"].reshape(-1), theta[0]))
        worst = max(worst, report("roi", ref["NNInput"] + AVG, roi[0].transpose(1, 2, 0)))
        worst = max(worst, report("feature", ref["fc1R"].reshape(-1), feat[0]))
        worst = max(worst, report("logits", ref["pred"].reshape(-1), logits[0]))

    # 3) batch + arbitrary input size
    xs56 = np.concatenate([nchw(load_rgb(f, 56)) for f in sample])
    xs300 = np.concatenate([nchw(load_rgb(f, 300)) for f in sample])
    feat_b, *_ = sess.run(None, {"input_56": xs56, "input_224": xs300})
    feat_1, *_ = sess.run(None, {"input_56": xs56[:1], "input_224": xs300[:1]})
    print(f"batch=3, input_224 at 300x300: feature {feat_b.shape}, |batch-single| = {np.abs(feat_b[0] - feat_1[0]).max():.2e}")

    # 4) recognition on NTU-PI-demo (gallery=train, probe=test)
    def embed(paths):
        x56 = np.concatenate([nchw(load_rgb(p, 56)) for p in paths])
        x224 = np.concatenate([nchw(load_rgb(p, 224)) for p in paths])
        feat, logits, _, roi = sess.run(None, {"input_56": x56, "input_224": x224})
        return feat, logits, roi

    gal = sorted(glob.glob(os.path.join(DEMO, "train", "*")))
    gf, gl, _ = embed(gal)
    pf, pl, proi = embed(files)
    gid = np.array([class_id(p) for p in gal])
    pid = np.array([class_id(p) for p in files])
    norm = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
    sim = norm(pf) @ norm(gf).T
    rank1 = (gid[sim.argmax(1)] == pid).mean()
    print(f"NTU-PI-demo: {len(files)} probes vs {len(gal)} gallery images, {len(set(gid))} palms")
    print(f"  rank-1 identification (cosine on 512-d feature): {rank1 * 100:.1f}%  (chance ~{100 / len(set(gid)):.1f}%)")
    same = np.equal.outer(pid, gid)
    print(f"  mean cosine same palm {sim[same].mean():.3f} vs different palm {sim[~same].mean():.3f}")
    pred = np.concatenate([gl, pl]).argmax(1) + 1  # classes.name are '1'..'2035'
    allid = np.concatenate([gid, pid])
    print(f"  classifier argmax == file Class id: {(pred == allid).mean() * 100:.1f}% of {len(allid)} images")

    out = os.path.join(ROOT, "output", "onnx_check_roi.jpg")
    tiles = [Image.fromarray(np.clip(r.transpose(1, 2, 0), 0, 255).astype(np.uint8)) for r in proi[:10]]
    canvas = Image.new("RGB", (112 * len(tiles), 112 * 2))
    for i, (t, p) in enumerate(zip(tiles, files[:10])):
        canvas.paste(Image.open(p).convert("RGB").resize((112, 112)), (112 * i, 0))
        canvas.paste(t, (112 * i, 112))
    canvas.save(out)
    print(f"  ROI preview (top: input, bottom: ROI from ONNX): {out}")
    print(f"worst relative difference vs reference: {worst:.1e} -> {'OK' if worst < 1e-3 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
