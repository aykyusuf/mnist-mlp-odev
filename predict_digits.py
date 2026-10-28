"""Kendi el yazisi okul numarasini MNIST formatina cevirip modele verir.

Kullanim:
  .venv/bin/python predict_digits.py my_digits/numara.png --label 123456789
  (--label verilirse tahminler gercek rakamlarla karsilastirilir)

Onisleme (MNIST'e benzetme):
  gri ton -> arka plan acik ise renk ters (MNIST: siyah zemin, beyaz rakam) -> rakamlari
  sutun bosluklarindan ayir -> her rakami 20x20'ye en-boy oranini bozmadan sigdir ->
  agirlik merkezine gore 28x28'in ortasina yerlestir -> /255 ve (x-0.1307)/0.3081 normalize.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from mnist_mlp import MEAN, STD, make_mlp

OUT = Path("outputs")


def to_mnist_gray(path):
    """Gri ton, [0,255], zemin siyah (0) rakam beyaz (255)."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA"):  # seffaf PNG'leri beyaz zemine yapistir
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img.convert("RGBA"))
    a = np.asarray(img.convert("L"), dtype=np.float32)
    # Zemin = kenar piksellerinin medyani; aciksa (kagit/beyaz tuval) ters cevir
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    if np.median(border) > 127:
        a = 255 - a
    a -= a.min()
    if a.max() > 0:
        a *= 255 / a.max()
    return a


def split_digits(a, min_gap=None):
    """Sutun projeksiyonu ile rakamlari ayirir; (x0, x1) listesi doner."""
    mask = a > 0.25 * 255
    cols = mask.any(axis=0)
    h, w = a.shape
    min_gap = min_gap or max(3, w // 100)
    segs, start, gap = [], None, 0
    for x in range(w):
        if cols[x]:
            if start is None:
                start = x
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                segs.append((start, x - gap + 1))
                start, gap = None, 0
    if start is not None:
        segs.append((start, w))
    return segs


def force_split(a, segs, n):
    """Birbirine degen rakamlar tek parca gelirse: en genis parcayi, sutun murekkebinin
    en az oldugu noktadan (orta %30-70 bandinda) bolerek toplam n rakama tamamla."""
    ink = (a > 0.25 * 255).sum(axis=0)
    segs = list(segs)
    while len(segs) < n:
        i = max(range(len(segs)), key=lambda k: segs[k][1] - segs[k][0])
        x0, x1 = segs[i]
        lo, hi = x0 + int(0.3 * (x1 - x0)), x0 + int(0.7 * (x1 - x0))
        cut = lo + int(np.argmin(ink[lo:hi + 1]))
        segs[i:i + 1] = [(x0, cut), (cut, x1)]
    return segs


def digit_to_28(a, x0, x1):
    """Tek rakami MNIST gibi 20x20 kutuya sigdirip agirlik merkeziyle 28x28'e ortalar."""
    mask = a[:, x0:x1] > 0.25 * 255
    ys = np.where(mask.any(axis=1))[0]
    crop = a[ys[0]:ys[-1] + 1, x0:x1]
    h, w = crop.shape
    s = 20 / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    small = np.asarray(Image.fromarray(crop.astype(np.uint8)).resize((nw, nh), Image.LANCZOS), dtype=np.float32)
    canvas = np.zeros((28, 28), dtype=np.float32)
    canvas[(28 - nh) // 2:(28 - nh) // 2 + nh, (28 - nw) // 2:(28 - nw) // 2 + nw] = small
    # agirlik merkezini 14,14'e kaydir (MNIST ayni sekilde ortalanmis)
    tot = canvas.sum()
    if tot > 0:
        cy = (canvas.sum(1) * np.arange(28)).sum() / tot
        cx = (canvas.sum(0) * np.arange(28)).sum() / tot
        dy, dx = int(round(14 - cy)), int(round(14 - cx))
        canvas = np.roll(canvas, (dy, dx), axis=(0, 1))
    return canvas


def load_model(path, hidden, dropout=0.0):
    m = make_mlp(hidden, dropout)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--label", help="gercek numara, orn. 123456789")
    ap.add_argument("--min-gap", type=int, default=None, help="rakamlar arasi min bosluk (px)")
    args = ap.parse_args()

    a = to_mnist_gray(args.image)
    segs = split_digits(a, args.min_gap)
    if args.label and len(segs) < len(args.label):
        print(f"{len(segs)} parca bulundu (rakamlar birbirine degiyor), {len(args.label)} rakama bolunuyor.")
        segs = force_split(a, segs, len(args.label))
    digits = np.stack([digit_to_28(a, x0, x1) for x0, x1 in segs])  # (N,28,28), 0-255
    print(f"{len(segs)} rakam bulundu.")
    if args.label and len(args.label) != len(segs):
        print(f"UYARI: beklenen {len(args.label)} rakam, bulunan {len(segs)}. --min-gap ile oynayin.")

    x = torch.from_numpy(digits / 255.0).float().sub(MEAN).div(STD).view(len(digits), -1)

    # models: (ad, agirlik dosyasi, hidden, dropout) - mnist_mlp.py CONFIGS ile ayni olmali
    import json
    res = json.load(open(OUT / "results.json"))
    best_cfg = next(r["cfg"] for r in res["results"] if r["cfg"]["name"] == res["best_by_validation"])
    models = {
        "baseline": load_model(OUT / "baseline_model.pt", [256, 128]),
        f"best({best_cfg['name']})": load_model(OUT / "best_model.pt", best_cfg["hidden"], best_cfg.get("dropout", 0.0)),
    }

    preds = {}
    with torch.no_grad():
        for name, m in models.items():
            p = torch.softmax(m(x), 1)
            preds[name] = (p.argmax(1).numpy(), p.max(1).values.numpy())

    for name, (pr, conf) in preds.items():
        s = "".join(map(str, pr))
        line = f"{name:22s} tahmin: {s}"
        if args.label and len(args.label) == len(pr):
            acc = np.mean([int(c) == p for c, p in zip(args.label, pr)])
            line += f"  | gercek: {args.label} | dogruluk: {acc:.0%}"
        print(line)
        print("  guven:", " ".join(f"{c:.2f}" for c in conf))

    # Gorsel: 28x28 onislenmis rakamlar + tahminler
    n = len(digits)
    fig, ax = plt.subplots(1, n, figsize=(1.3 * n, 1.9), squeeze=False)
    first = next(iter(preds.values()))
    for i in range(n):
        ax[0, i].imshow(digits[i], cmap="gray")
        t = f"{first[0][i]}"
        if args.label and len(args.label) == n:
            t = f"{args.label[i]}->{first[0][i]}"
        ax[0, i].set_title(t, fontsize=9, color="green" if (not args.label or len(args.label) != n or int(args.label[i]) == first[0][i]) else "red")
        ax[0, i].axis("off")
    fig.suptitle(f"Onislenmis 28x28 ({next(iter(preds))} tahmini)", fontsize=9)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    out = OUT / f"preprocessed_{Path(args.image).stem}.png"
    fig.savefig(out, dpi=130)
    print("kaydedildi:", out)


if __name__ == "__main__":
    main()
