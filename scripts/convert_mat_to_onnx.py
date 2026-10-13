"""Convert the MatConvNet (DagNN) palmprint models to ONNX.

Models (Matkowski et al., "Palmprint Recognition in Uncontrolled and
Uncooperative Environment", IEEE TIFS 2019):
  models/AlignNet-epoch-25.mat  -> models/AlignNet-epoch-25.onnx   (ROI-LAnet)
  models/EE_PRnet-epoch-60.mat  -> models/EE_PRnet-epoch-60.onnx   (EE-PRnet)

The .mat files are walked layer by layer and rebuilt in PyTorch with the
exact MatConvNet semantics (see Palmprint-Recognition-in-the-Wild-master/
functions/buildFeat.m and ROI-LAnet/demo3.m), then exported with
torch.onnx. Loss layers are dropped, dropout is identity (test mode).

ONNX interface (all images: float32, NCHW, RGB, pixel range 0..255; the
ImageNet mean [123.68 116.78 103.94] is subtracted inside the graph).
Resize the way vl_imreadjpeg(...,'Resize',[H W]) does (training and
buildFeat.m): centred crop with the output aspect ratio, then a float
triangle filter widened for downsampling, borders replicated
(palm-enroll/python/vl_imread.py, palm-enroll/src/preprocess.cpp).

  AlignNet-epoch-25.onnx
    in  input_56    [N,3,56,56]   hand image resized to 56x56
    in  input_full  [N,3,H,W]     same hand image, any size (sampled by TPS)
    out theta       [N,18]        TPS control points, [x1..x9, y1..y9] in [-1,1]
    out roi         [N,3,112,112] aligned palm ROI, RGB 0..255 (outside the
                                  image: mean colour; demo3.m gives black)

  EE_PRnet-epoch-60.onnx
    in  input_56    [N,3,56,56]   hand image resized to 56x56
    in  input_224   [N,3,H,W]     same hand image resized to 224x224, as in
                                  training (getBatchNclassSpecial.m). Any size
                                  is accepted: buildFeat.m 'imgOrg' feeds the
                                  original size, but meta.inuptType 'imgOrg'
                                  only names the image folder used to train
    out feature     [N,512]       fc1R embedding (the recognition feature)
    out logits      [N,2035]      NTU-PI-v1 class scores (training classes)
    out theta       [N,18]
    out roi         [N,3,112,112]

Usage:
  venv/bin/python scripts/convert_mat_to_onnx.py [--opset 17]
"""
import argparse
import os

import numpy as np
import onnx
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AVERAGE_IMAGE = (123.68, 116.78, 103.94)  # buildFeat.m: avimnet

MODELS = {
    "AlignNet-epoch-25": {
        "inputs": {"ANInput": "input_56"},
        # ROI-LAnet/demo3.m appends grid generator + sampler (112x112)
        "append_sampler": {"theta": "theta", "image": "input_full", "size": 112},
        "outputs": {"theta": "theta", "NNInput": "roi"},
    },
    "EE_PRnet-epoch-60": {
        "inputs": {"ANInput": "input_56", "Input224": "input_224"},
        "outputs": {"fc1R": "feature", "pred": "logits", "theta": "theta", "NNInput": "roi"},
    },
}
DYNAMIC_HW_INPUTS = {"input_full", "input_224"}
SKIP_LAYER_TYPES = {"dagnn.Loss", "dagnn.PDist"}


def as_list(x):
    if isinstance(x, str):
        return [x]
    return [] if x is None else list(np.atleast_1d(x))


def tps_sampling_matrix(block):
    """dagnnExtra.TpsGridGenerator is linear in theta.

    XGp = M @ Xp, YGp = M @ Yp with M = [1 XG YG] Li_a + U(|G - CP|) Li_w,
    U(r) = r^2 log r^2, U(0) = 0  (matlab/+dagnnExtra/TpsGridGenerator.m).
    Returns M of shape [Ho*Wo, Nkp].
    """
    XG = np.asarray(block.XG, np.float64).reshape(-1)
    YG = np.asarray(block.YG, np.float64).reshape(-1)
    X = np.asarray(block.X, np.float64).reshape(-1)
    Y = np.asarray(block.Y, np.float64).reshape(-1)
    Li_w = np.asarray(block.Li_w, np.float64)
    Li_a = np.asarray(block.Li_a, np.float64)
    r2 = (XG[:, None] - X[None]) ** 2 + (YG[:, None] - Y[None]) ** 2
    U = np.where(r2 == 0, 0.0, r2 * np.log(np.where(r2 == 0, 1.0, r2)))
    P = np.stack([np.ones_like(XG), XG, YG], axis=1)
    return (P @ Li_a + U @ Li_w).astype(np.float32)


def tps_block_from_theta(n_theta, size):
    """Build a TpsGridGenerator('Ho',size,'Wo',size,'k',3,'lambda',0)."""
    k = int(round(np.sqrt(n_theta // 2)))
    X, Y = np.meshgrid(np.linspace(-1, 1, k), np.linspace(-1, 1, k))
    X, Y = X.reshape(-1, order="F"), Y.reshape(-1, order="F")  # matlab X(:)
    r2 = (X[:, None] - X[None]) ** 2 + (Y[:, None] - Y[None]) ** 2
    K = np.where(r2 == 0, 0.0, r2 * np.log(np.where(r2 == 0, 1.0, r2)))
    P = np.stack([np.ones_like(X), X, Y], axis=1)
    L = np.block([[K, P], [P.T, np.zeros((3, 3))]])
    Li = np.linalg.inv(L)
    XG, YG = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))

    class Block:
        pass

    b = Block()
    b.X, b.Y = X.astype(np.float32), Y.astype(np.float32)
    b.Li_w = Li[: k * k, : k * k].astype(np.float32)
    b.Li_a = Li[k * k :, : k * k].astype(np.float32)
    b.XG, b.YG, b.Ho, b.Wo = XG.astype(np.float32), YG.astype(np.float32), size, size
    return b


class Conv(nn.Module):
    def __init__(self, block, weight, bias):
        super().__init__()
        w = weight.reshape(tuple(int(s) for s in block.size))  # [h,w,cin,cout]
        self.weight = nn.Parameter(torch.from_numpy(np.ascontiguousarray(w.transpose(3, 2, 0, 1))))
        self.bias = None if bias is None else nn.Parameter(torch.from_numpy(bias.reshape(-1).copy()))
        t, b, l, r = (int(p) for p in np.broadcast_to(block.pad, 4))
        self.symmetric = t == b and l == r
        self.pad = (t, l) if self.symmetric else (l, r, t, b)
        self.stride = tuple(int(s) for s in np.broadcast_to(block.stride, 2))
        self.dilation = tuple(int(d) for d in np.broadcast_to(block.dilate, 2))
        self.out_channels = w.shape[3]

    def forward(self, x):
        if self.symmetric:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.pad, self.dilation)
        return F.conv2d(F.pad(x, self.pad), self.weight, self.bias, self.stride, 0, self.dilation)


class MaxPool(nn.Module):
    def __init__(self, block):
        super().__init__()
        if block.method != "max":
            raise NotImplementedError(f"pooling method {block.method}")
        t, b, l, r = (int(p) for p in np.broadcast_to(block.pad, 4))
        self.pad = (l, r, t, b)
        self.kernel = tuple(int(s) for s in np.broadcast_to(block.poolSize, 2))
        self.stride = tuple(int(s) for s in np.broadcast_to(block.stride, 2))
        # MatConvNet's usual [0 s-1 0 s-1] padding with kernel == stride is ceil_mode
        self.ceil = (t, l) == (0, 0) and self.kernel == self.stride and (b, r) == tuple(s - 1 for s in self.stride)

    def forward(self, x):
        if self.ceil:
            return F.max_pool2d(x, self.kernel, self.stride, ceil_mode=True)
        if any(self.pad):
            x = F.pad(x, self.pad, value=float("-inf"))
        return F.max_pool2d(x, self.kernel, self.stride)


class LRN(nn.Module):
    """vl_nnnormalize: y = x / (kappa + alpha * sum_{q in Q(k)} x_q^2)^beta,
    Q(k) = [k - floor((N-1)/2), k + ceil((N-1)/2)] (alpha NOT divided by N)."""

    def __init__(self, block, channels):
        super().__init__()
        n, self.kappa, self.alpha, self.beta = (float(p) for p in block.param)
        lo, hi = int(np.floor((n - 1) / 2)), int(np.ceil((n - 1) / 2))
        k = np.arange(channels)
        window = ((k[None] >= k[:, None] - lo) & (k[None] <= k[:, None] + hi)).astype(np.float32)
        self.full = bool(window.all())
        self.register_buffer("window", torch.from_numpy(window)[:, :, None, None])

    def forward(self, x):
        sq = x * x
        s = sq.sum(1, keepdim=True) if self.full else F.conv2d(sq, self.window)
        return x / torch.pow(self.kappa + self.alpha * s, self.beta)


class TpsGrid(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.ho, self.wo = int(block.Ho), int(block.Wo)
        self.register_buffer("M", torch.from_numpy(tps_sampling_matrix(block)))
        self.nkp = self.M.shape[1]

    def forward(self, theta):
        theta = theta.flatten(1)  # matlab 1x1x(2K)xN -> [X'_1..X'_K, Y'_1..Y'_K]
        xg = (theta[:, : self.nkp] @ self.M.t()).view(-1, self.ho, self.wo)
        yg = (theta[:, self.nkp :] @ self.M.t()).view(-1, self.ho, self.wo)
        return torch.stack([xg, yg], dim=-1)  # grid_sample wants (x, y)


class BilinearSampler(nn.Module):
    """vl_nnbilinearsampler: (-1,-1)/(+1,+1) are the centres of the corner
    pixels (align_corners=True), samples outside the image are zero."""

    def forward(self, x, grid):
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


class LeakyReLU(nn.Module):
    def __init__(self, leak):
        super().__init__()
        self.leak = float(leak)

    def forward(self, x):
        return F.relu(x) if self.leak == 0 else F.leaky_relu(x, self.leak)


class DagNet(nn.Module):
    def __init__(self, net, cfg):
        super().__init__()
        params = {p.name: np.asarray(p.value, np.float32) for p in net.params}
        self.input_vars = list(cfg["inputs"])
        self.output_vars = list(cfg["outputs"])
        self.register_buffer("mean", torch.tensor(AVERAGE_IMAGE, dtype=torch.float32).view(1, 3, 1, 1))
        channels = {v: 3 for v in self.input_vars}
        self.graph = []  # (module name, input vars, output var)
        self.ops = nn.ModuleDict()

        def add(name, module, ins, out):
            key = name.replace(".", "_")
            self.ops[key] = module
            self.graph.append((key, ins, out))

        for L in np.atleast_1d(net.layers):
            if L.type in SKIP_LAYER_TYPES:
                continue
            ins, outs, pnames = as_list(L.inputs), as_list(L.outputs), as_list(L.params)
            blk, out = L.block, outs[0]
            if L.type == "dagnn.Conv":
                bias = params[pnames[1]] if len(pnames) > 1 else None
                m = Conv(blk, params[pnames[0]], bias)
                channels[out] = m.out_channels
            elif L.type == "dagnn.ReLU":
                m = LeakyReLU(blk.leak)
            elif L.type == "dagnn.Pooling":
                m = MaxPool(blk)
            elif L.type == "dagnn.LRN":
                m = LRN(blk, channels[ins[0]])
            elif L.type == "dagnn.DropOut":
                m = nn.Identity()
            elif L.type == "dagnnExtra.TpsGridGenerator":
                m = TpsGrid(blk)
            elif L.type == "dagnn.BilinearSampler":
                m = BilinearSampler()
                channels[out] = channels[ins[0]]
            else:
                raise NotImplementedError(f"layer {L.name}: {L.type}")
            channels.setdefault(out, channels.get(ins[0]))
            add(L.name, m, ins, out)

        app = cfg.get("append_sampler")
        if app:
            tps = tps_block_from_theta(channels[app["theta"]], app["size"])
            add("grid_generator", TpsGrid(tps), [app["theta"]], "grid")
            add("sampler", BilinearSampler(), ["Input224", "grid"], "NNInput")
            self.input_vars.append("Input224")

    def forward(self, *images):
        v = {name: img - self.mean for name, img in zip(self.input_vars, images)}
        for key, ins, out in self.graph:
            v[out] = self.ops[key](*[v[i] for i in ins])
        res = []
        for name in self.output_vars:
            y = v[name]
            if name == "NNInput":
                y = y + self.mean  # ROI back to RGB 0..255
            elif y.dim() == 4 and y.shape[2:] == (1, 1):
                y = y.flatten(1)
            res.append(y)
        return tuple(res)


def load_net(path):
    return sio.loadmat(path, struct_as_record=False, squeeze_me=True)["net"]


def build(name):
    cfg = MODELS[name]
    model = DagNet(load_net(os.path.join(ROOT, "models", name + ".mat")), cfg).eval()
    in_names = list(cfg["inputs"].values())
    if cfg.get("append_sampler"):
        in_names.append(cfg["append_sampler"]["image"])
    return model, in_names, list(cfg["outputs"].values())


def export(name, opset):
    model, in_names, out_names = build(name)
    shapes = {"input_56": (1, 3, 56, 56), "input_224": (1, 3, 224, 224), "input_full": (1, 3, 224, 224)}
    dummy = tuple(torch.rand(*shapes[n]) * 255 for n in in_names)
    dyn = {n: {0: "batch"} for n in in_names + out_names}
    for n in in_names:
        if n in DYNAMIC_HW_INPUTS:
            dyn[n] = {0: "batch", 2: "height", 3: "width"}
    dst = os.path.join(ROOT, "models", name + ".onnx")
    torch.onnx.export(
        model, dummy, dst, input_names=in_names, output_names=out_names,
        dynamic_axes=dyn, opset_version=opset, dynamo=False,
    )
    # the exporter loses the static output sizes behind GridSample; write them back
    proto = onnx.load(dst)
    for value, ref in zip(proto.graph.output, model(*dummy)):
        for dim, size in zip(value.type.tensor_type.shape.dim[1:], ref.shape[1:]):
            dim.Clear()
            dim.dim_value = size
    onnx.checker.check_model(proto)
    onnx.save(proto, dst)
    print(f"{name}: {dst} ({os.path.getsize(dst) / 2**20:.1f} MB) inputs={in_names} outputs={out_names}")
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    args = ap.parse_args()
    with torch.no_grad():
        for name in args.models:
            export(name, args.opset)


if __name__ == "__main__":
    main()
