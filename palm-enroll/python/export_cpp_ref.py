"""Reference data for tests/test_parity.cpp.

For a few images it stores the input as PNG (lossless, so JPEG decoders do not
matter), the network inputs resized the vl_imreadjpeg way (vl_imread.py) and
the outputs of EE_PRnet-epoch-60.onnx run by onnxruntime in Python.
verify_onnx.py ties that ONNX model to the MatConvNet forward pass, so the
C++ test closes the chain MatConvNet -> ONNX -> C++.

Usage (from palm-enroll/):
  ../venv/bin/python python/export_cpp_ref.py
"""
import csv
import os

import numpy as np
import onnxruntime as ort
from PIL import Image

from vl_imread import vl_resize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "..", "tests", "data", "ref")
ONNX = os.path.join(ROOT, "models", "EE_PRnet-epoch-60.onnx")
FLIP_HANDEDNESS = "Right"  # mirrored in the reference so the C++ flip path is tested


def cases():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "log_label.csv"), encoding="utf-8"), delimiter=";"))
    hands, people, count = [], set(), {"Left": 0, "Right": 0}
    for r in rows:  # two Left and two Right hand crops, all from different people
        path = os.path.join(ROOT, "output", "hand_crop", r["filename"] + ".jpg")
        if count.get(r["handedness"], 2) < 2 and r["person_id"] not in people and os.path.exists(path):
            people.add(r["person_id"])
            count[r["handedness"]] += 1
            hands.append((path, r["handedness"] == FLIP_HANDEDNESS))
    for i, (path, flip) in enumerate(hands):
        # downscaled to keep the test data small; the PNG itself is the reference input
        yield f"hand{i}", Image.open(path).convert("RGB").resize((320, 320), Image.LANCZOS), flip
    full = sorted(os.listdir(os.path.join(ROOT, "output", "original")))[0]  # non-square: crop path
    yield "frame", Image.open(os.path.join(ROOT, "output", "original", full)).convert("RGB").resize((480, 270)), False
    demo = os.path.join(ROOT, "Palmprint-Recognition-in-the-Wild-master/databases/NTU-PI-demo/flip/img/test")
    yield "ntu", Image.open(os.path.join(demo, sorted(os.listdir(demo))[0])).convert("RGB"), False


def main():
    os.makedirs(OUT, exist_ok=True)
    sess = ort.InferenceSession(ONNX)
    lines = []
    for name, im, flip in cases():
        im.save(os.path.join(OUT, name + ".png"))
        rgb = np.asarray(im, np.float32)
        if flip:
            rgb = rgb[:, ::-1]
        x56, x224 = vl_resize(rgb, 56, 56), vl_resize(rgb, 224, 224)
        chw = lambda a: np.ascontiguousarray(a.transpose(2, 0, 1)[None])
        feat, theta, roi = sess.run(["feature", "theta", "roi"], {"input_56": chw(x56), "input_224": chw(x224)})
        for suffix, arr in (("x56", x56), ("x224", x224), ("feature", feat), ("theta", theta), ("roi", roi)):
            np.ascontiguousarray(arr, np.float32).tofile(os.path.join(OUT, f"{name}.{suffix}.f32"))
        lines.append(f"{name} {int(flip)}")
        print(f"{name}: {im.size[0]}x{im.size[1]} flip={int(flip)} |feature|={np.linalg.norm(feat):.3f}")
    with open(os.path.join(OUT, "manifest.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("written to", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
