#!/usr/bin/env python3
"""Train GG-CNN for Mode 2 grasp detection and export it to ONNX.

Two-step pipeline (keeps slow raw-dataset parsing out of the train loop):

    1. python3 tools/prepare_dataset.py --dataset cornell --data <raw> --out data/prep
    2. python3 tools/train_ggcnn.py    --data data/prep --epochs 40 \
           --export-onnx models/ggcnn.onnx

``prepare_dataset.py`` renders Cornell/Jacquard into cached ``.npz`` shards
(``input`` CxHxW + ``target`` 4xHxW); this script just reads them. The exported
model emits four out_size x out_size maps in the order [quality, cos2θ, sin2θ,
width] — exactly what gripper/vision/grasp_vision.py's GGCNNGraspDetector +
decode_ggcnn consume (ONNX Runtime, no torch on the Pi).

The label encoder ``draw_grasp_target`` (used by the prepare step) is the inverse
of decode_ggcnn and is pure + unit-tested. The model / training / export are
torch-gated and lazy-imported, so this file imports without torch installed.
Heavy work needs torch + a GPU; it runs at your desk, not on the Pi.
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys
from typing import Sequence, Tuple

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

Grasp = Tuple[float, float, float, float]  # (cx, cy, theta_deg, width_px) in map space


# ---------------------------------------------------------------------------
# Pure label encoding — the inverse of decode_ggcnn (unit-tested, numpy only).
# Shared with tools/prepare_dataset.py, which renders the dataset target maps.
# ---------------------------------------------------------------------------

def draw_grasp_target(out_size: int, grasps: Sequence[Grasp],
                      width_scale: float = 150.0, radius: int = 8):
    """Render labelled grasps into the four GG-CNN target maps.

    Each grasp paints a filled disk of ``radius`` px at its centre into the
    quality map (1.0), with the angle written as (cos2θ, sin2θ) and the jaw width
    as ``width_px / width_scale`` — matching how ``decode_ggcnn`` reads them back.
    Returns ``(quality, cos2, sin2, width)`` as float32 arrays.
    """
    import numpy as np

    q = np.zeros((out_size, out_size), dtype=np.float32)
    cos2 = np.zeros_like(q)
    sin2 = np.zeros_like(q)
    width = np.zeros_like(q)
    yy, xx = np.mgrid[0:out_size, 0:out_size]

    for cx, cy, theta_deg, w_px in grasps:
        th = math.radians(theta_deg)
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
        q[disk] = 1.0
        cos2[disk] = math.cos(2 * th)
        sin2[disk] = math.sin(2 * th)
        width[disk] = w_px / width_scale
    return q, cos2, sin2, width


# ---------------------------------------------------------------------------
# torch-gated: dataset reader, model, train loop, ONNX export (lazy imports)
# ---------------------------------------------------------------------------

def _require_torch():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"missing '{e.name}' — training needs PyTorch on a workstation: "
            "pip install torch torchvision  (the Pi only runs the exported .onnx)")


def build_dataset(root: str):
    """Read the prepared ``.npz`` shards listed in ``<root>/manifest.txt``."""
    import numpy as np
    from torch.utils.data import Dataset

    manifest = os.path.join(root, "manifest.txt")
    if not os.path.exists(manifest):
        raise SystemExit(
            f"no manifest.txt in {root} — run tools/prepare_dataset.py first, e.g. "
            f"python3 tools/prepare_dataset.py --dataset cornell --data <raw> --out {root}")
    with open(manifest) as fh:
        names = [ln.strip() for ln in fh if ln.strip()]

    class PreparedDataset(Dataset):
        def __init__(self):
            self.files = [os.path.join(root, n) for n in names]
            self.channels = int(np.load(self.files[0])["input"].shape[0])

        def __len__(self):
            return len(self.files)

        def __getitem__(self, i):
            d = np.load(self.files[i])
            return d["input"].astype(np.float32), d["target"].astype(np.float32)

    return PreparedDataset()


def build_model(input_channels: int):
    import torch.nn as nn

    class GGCNN(nn.Module):
        """Fully-convolutional GG-CNN (Morrison et al.). out_size must be divisible
        by 12 (3*2*2) — 300 is. 1x1 output heads keep the maps exactly out_size."""

        def __init__(self, c: int):
            super().__init__()
            self.conv1 = nn.Conv2d(c, 32, 9, stride=3, padding=3)
            self.conv2 = nn.Conv2d(32, 16, 5, stride=2, padding=2)
            self.conv3 = nn.Conv2d(16, 8, 3, stride=2, padding=1)
            self.convt1 = nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1, output_padding=1)
            self.convt2 = nn.ConvTranspose2d(8, 16, 5, stride=2, padding=2, output_padding=1)
            self.convt3 = nn.ConvTranspose2d(16, 32, 9, stride=3, padding=3, output_padding=0)
            self.pos = nn.Conv2d(32, 1, 1)
            self.cos = nn.Conv2d(32, 1, 1)
            self.sin = nn.Conv2d(32, 1, 1)
            self.width = nn.Conv2d(32, 1, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.relu(self.convt1(x))
            x = self.relu(self.convt2(x))
            x = self.relu(self.convt3(x))
            return self.pos(x), self.cos(x), self.sin(x), self.width(x)

    return GGCNN(input_channels)


def train(args) -> None:
    _require_torch()
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = build_dataset(args.data)
    channels = ds.channels
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    model = build_model(channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"{len(ds)} samples, {channels}-channel input, device={device}")

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for inp, target in dl:
            inp = inp.to(device)
            pos_t, cos_t, sin_t, w_t = (target[:, k:k + 1].to(device) for k in range(4))
            pos_p, cos_p, sin_p, w_p = model(inp)
            loss = (F.mse_loss(pos_p, pos_t) + F.mse_loss(cos_p, cos_t)
                    + F.mse_loss(sin_p, sin_t) + F.mse_loss(w_p, w_t))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
        print(f"epoch {epoch + 1}/{args.epochs}  loss {total / max(1, len(dl)):.4f}")

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved weights -> {args.out}")
    if args.export_onnx:
        export_onnx(model, channels, args.out_size, args.export_onnx)


def export_onnx(model, channels: int, out_size: int, path: str) -> None:
    import torch

    model.eval()
    dummy = torch.zeros(1, channels, out_size, out_size)
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, path, opset_version=12,
        input_names=["input"],
        output_names=["pos", "cos", "sin", "width"],  # decode_ggcnn order
        dynamic_axes={"input": {0: "batch"}},
    )
    print(f"exported ONNX -> {path}  (outputs: pos, cos, sin, width)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="prepared dataset dir (manifest.txt)")
    ap.add_argument("--out", default="models/ggcnn.pt", help="weights output")
    ap.add_argument("--export-onnx", default="models/ggcnn.onnx",
                    help="also export ONNX here ('' to skip)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-size", type=int, default=300,
                    help="square map side for ONNX export (must match prepare)")
    args = ap.parse_args()
    if args.out_size % 12:
        ap.error("--out-size must be divisible by 12 (e.g. 300)")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
