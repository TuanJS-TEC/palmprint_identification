"""Group an ONNX Runtime profile of EE-PRnet (palm_profile --ort-profile) by
pipeline step.

Node names come from convert_mat_to_onnx.py; ORT's NCHWc transformer renames
a fused Conv+activation after its output (e.g. /NNrelu1_2/Relu_output_0_nchwc)
and inserts Reorder* layout nodes.

Usage:
  python ort_profile_groups.py PROFILE.json [--skip 5]
"""
import argparse
import collections
import json
import re

GROUPS = [  # (step, regex on the node name), first match wins
    ("normalisation: mean subtraction", r"^/Sub(_1)?$"),
    ("alignment: ROI-LAnet (theta regression)", r"^/(AN|featNormPool|relu1/|relu2/|fcr3_4/)"),
    ("alignment: TPS grid from theta", r"^/aff/"),
    ("crop ROI: TPS sampling 112x112", r"^/samp/"),
    ("extract vector: recognition CNN + fc1R", r"^/(NN|FeatNormR|fcr1_4R)"),
    ("unused: NTU-PI logits head", r"^/(fcr2_4R|Flatten_1$)"),
    ("outputs: roi (+mean), flatten", r"^/(Add$|Flatten)"),
    ("layout: NCHWc reorder", r"^Reorder"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--skip", type=int, default=5, help="first runs left out (arena warm-up)")
    args = ap.parse_args()

    events = json.load(open(args.profile))
    runs = sorted((e for e in events if e.get("cat") == "Session" and e["name"] == "model_run"), key=lambda e: e["ts"])
    kept = runs[args.skip:]
    if not kept:
        raise SystemExit("no model_run left after --skip")
    t0 = kept[0]["ts"]

    per_group = collections.defaultdict(float)  # us, summed over the kept runs
    unknown = collections.Counter()
    for e in events:
        if e.get("cat") != "Node" or not e["name"].endswith("_kernel_time") or e["ts"] < t0:
            continue
        node = e["name"][: -len("_kernel_time")]
        step = next((g for g, rx in GROUPS if re.search(rx, node)), None)
        if step is None:
            unknown[node] += e["dur"]
            step = "other"
        per_group[step] += e["dur"]

    n = len(kept)
    run_ms = sum(e["dur"] for e in kept) / n / 1000
    nodes_ms = sum(per_group.values()) / n / 1000
    print(f"{n} runs (first {args.skip} skipped): model_run {run_ms:.3f} ms, kernels {nodes_ms:.3f} ms")
    print(f"  {'step':44s} {'ms/run':>8s} {'% run':>6s}")
    for g, _ in GROUPS + [("other", None)]:
        if g in per_group:
            ms = per_group[g] / n / 1000
            print(f"  {g:44s} {ms:8.3f} {100 * ms / run_ms:5.1f}%")
    ms = run_ms - nodes_ms
    print(f"  {'executor overhead (model_run - kernels)':44s} {ms:8.3f} {100 * ms / run_ms:5.1f}%")
    for node, us in unknown.most_common():
        print(f"  other: {node} {us / n / 1000:.3f} ms")


if __name__ == "__main__":
    main()
