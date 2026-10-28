"""Onisleme adimlari olmadan el yazisi numarasinin tahmini (README'deki ablation tablosu).
Calistir: .venv/bin/python ablation_preprocess.py   (once mnist_mlp.py calismis olmali)"""
import json

import numpy as np
import torch
from PIL import Image

from mnist_mlp import MEAN, STD, make_mlp
from predict_digits import digit_to_28, force_split, split_digits, to_mnist_gray

LABEL = "23120205021"
a = to_mnist_gray("my_digits/numara.jpg")
segs = force_split(a, split_digits(a), len(LABEL))

res = json.load(open("outputs/results.json"))
best = next(r["cfg"] for r in res["results"] if r["cfg"]["name"] == res["best_by_validation"])
models = {"baseline": make_mlp([256, 128]), best["name"]: make_mlp(best["hidden"])}
models["baseline"].load_state_dict(torch.load("outputs/baseline_model.pt"))
models[best["name"]].load_state_dict(torch.load("outputs/best_model.pt"))


def predict(digits):
    x = torch.from_numpy(digits / 255.0).float().sub(MEAN).div(STD).view(len(digits), -1)
    out = {}
    for k, m in models.items():
        p = "".join(map(str, m.eval()(x).argmax(1).tolist()))
        out[k] = (p, sum(c == d for c, d in zip(LABEL, p)))
    return out


full = np.stack([digit_to_28(a, x0, x1) for x0, x1 in segs])
stretched = []
for x0, x1 in segs:  # kirp, 20x20'ye sigdirmeden/ortalamadan dogrudan 28x28'e esnet
    ys = np.where((a[:, x0:x1] > 0.25 * 255).any(1))[0]
    crop = a[ys[0]:ys[-1] + 1, x0:x1].astype(np.uint8)
    stretched.append(np.asarray(Image.fromarray(crop).resize((28, 28), Image.LANCZOS), dtype=np.float32))

for name, d in [("tam onisleme", full), ("renk tersleme yok", 255 - full), ("20x20 + ortalama yok", np.stack(stretched))]:
    print(f"{name:22s}", {k: f"{p} ({n}/11)" for k, (p, n) in predict(d).items()})
