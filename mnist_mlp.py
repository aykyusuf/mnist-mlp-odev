"""MNIST uzerinde MLP: Deney 1 (baseline) + hiperparametre karsilastirmasi.

Calistir:  .venv/bin/python mnist_mlp.py
Cikti:     outputs/ (grafikler, results.json, baseline_model.pt, best_model.pt)

Kurallar: CNN yok, pretrained yok, model sifirdan egitilir.
Test seti sadece EN SONDA kullanilir (baseline + validation'a gore secilen en iyi config).
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets

SEED = 42
VAL_SIZE = 10_000          # 60k train -> 50k train + 10k validation
MEAN, STD = 0.1307, 0.3081  # MNIST train istatistikleri (predict_digits.py ayni degerleri kullanir)
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

device = torch.device("cpu")  # MLP kucuk, CPU yeterli ve tekrarlanabilir


def load_data():
    tr = datasets.MNIST("data", train=True, download=True)
    te = datasets.MNIST("data", train=False, download=True)

    def prep(ds):
        x = ds.data.float().div(255).sub(MEAN).div(STD).view(len(ds), -1)  # 28x28 -> 784
        return x, ds.targets

    x, y = prep(tr)
    x_te, y_te = prep(te)
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(x), generator=g)
    val_idx, tr_idx = perm[:VAL_SIZE], perm[VAL_SIZE:]
    return (x[tr_idx], y[tr_idx]), (x[val_idx], y[val_idx]), (x_te, y_te)


def make_mlp(hidden, dropout=0.0):
    layers, d = [], 784
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, 10))  # logits; softmax CrossEntropyLoss icinde
    return nn.Sequential(*layers)


@torch.no_grad()
def evaluate(model, x, y, loss_fn, bs=2000):
    model.eval()
    tot_loss, correct = 0.0, 0
    for i in range(0, len(x), bs):
        out = model(x[i:i + bs])
        tot_loss += loss_fn(out, y[i:i + bs]).item() * len(out)
        correct += (out.argmax(1) == y[i:i + bs]).sum().item()
    return tot_loss / len(x), correct / len(x)


def run(cfg, train, val):
    torch.manual_seed(SEED)
    model = make_mlp(cfg["hidden"], cfg.get("dropout", 0.0)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    if cfg["optimizer"] == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    else:
        opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg.get("momentum", 0.0))

    x_tr, y_tr = train
    hist = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
    g = torch.Generator().manual_seed(SEED)
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(x_tr), generator=g)
        for i in range(0, len(x_tr), cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            opt.zero_grad()
            loss_fn(model(x_tr[idx]), y_tr[idx]).backward()
            opt.step()
        # epoch sonu metrikleri (train metrigi eval modunda, tum train seti uzerinde)
        tl, ta = evaluate(model, *train, loss_fn)
        vl, va = evaluate(model, *val, loss_fn)
        for k, v in zip(hist, (tl, vl, ta, va)):
            hist[k].append(v)
        print(f"  [{cfg['name']}] ep {ep + 1:2d}/{cfg['epochs']}  "
              f"train {tl:.4f}/{ta:.4f}  val {vl:.4f}/{va:.4f}")
    return model, hist, time.time() - t0


def plot_history(hist, title, path):
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, hist["train_loss"], label="train")
    ax[0].plot(ep, hist["val_loss"], label="validation")
    ax[0].set(title=f"{title} - Loss", xlabel="epoch", ylabel="CrossEntropy loss")
    ax[1].plot(ep, hist["train_acc"], label="train")
    ax[1].plot(ep, hist["val_acc"], label="validation")
    ax[1].set(title=f"{title} - Accuracy", xlabel="epoch", ylabel="accuracy")
    for a in ax:
        a.grid(alpha=.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# Deney 1 = ilk config (baseline). Digerleri hiperparametre karsilastirmasi.
CONFIGS = [
    dict(name="baseline", hidden=[256, 128], optimizer="adam", lr=1e-3, batch_size=128, epochs=15),
    dict(name="kucuk_agi", hidden=[64, 32], optimizer="adam", lr=1e-3, batch_size=128, epochs=15),
    dict(name="buyuk_ag", hidden=[512, 256, 128], optimizer="adam", lr=1e-3, batch_size=128, epochs=15),
    dict(name="yuksek_lr", hidden=[256, 128], optimizer="adam", lr=1e-2, batch_size=128, epochs=15),
    dict(name="sgd_momentum", hidden=[256, 128], optimizer="sgd", lr=0.05, momentum=0.9, batch_size=128, epochs=15),
    dict(name="dropout0.3", hidden=[256, 128], optimizer="adam", lr=1e-3, batch_size=128, epochs=15, dropout=0.3),
]


def main():
    train, val, test = load_data()
    print(f"train {len(train[0])} | val {len(val[0])} | test {len(test[0])}")
    loss_fn = nn.CrossEntropyLoss()

    results, models, hists = [], {}, {}
    for cfg in CONFIGS:
        print(f"\n== {cfg['name']} {cfg}")
        model, hist, secs = run(cfg, train, val)
        models[cfg["name"]], hists[cfg["name"]] = model, hist
        plot_history(hist, cfg["name"], OUT / f"curves_{cfg['name']}.png")
        results.append(dict(
            cfg=cfg, train_time_sec=round(secs, 1),
            final_train_loss=hist["train_loss"][-1], final_val_loss=hist["val_loss"][-1],
            final_train_acc=hist["train_acc"][-1], final_val_acc=hist["val_acc"][-1],
            best_val_acc=max(hist["val_acc"]),
        ))

    # Karsilastirma grafigi (sadece validation)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for n, h in hists.items():
        ax[0].plot(range(1, len(h["val_loss"]) + 1), h["val_loss"], label=n)
        ax[1].plot(range(1, len(h["val_acc"]) + 1), h["val_acc"], label=n)
    ax[0].set(title="Validation loss", xlabel="epoch")
    ax[1].set(title="Validation accuracy", xlabel="epoch", ylim=(0.9, 1.0))
    for a in ax:
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "comparison_val.png", dpi=130)
    plt.close(fig)

    # Model secimi SADECE validation'a gore (son epoch val acc)
    best = max(results, key=lambda r: r["final_val_acc"])["cfg"]["name"]

    # TEST: yalniz baseline (Deney 1 icin zorunlu) ve val ile secilen en iyi model
    for r in results:
        n = r["cfg"]["name"]
        if n in ("baseline", best):
            _, r["test_acc"] = evaluate(models[n], *test, loss_fn)
    torch.save(models["baseline"].state_dict(), OUT / "baseline_model.pt")
    torch.save(models[best].state_dict(), OUT / "best_model.pt")

    json.dump(dict(seed=SEED, split=dict(train=len(train[0]), val=len(val[0]), test=len(test[0])),
                   best_by_validation=best, results=results),
              open(OUT / "results.json", "w"), indent=2, ensure_ascii=False)

    print(f"\nEn iyi (validation'a gore): {best}")
    print(f"{'config':14s} {'tr_loss':>8s} {'val_loss':>8s} {'tr_acc':>7s} {'val_acc':>7s} {'test':>7s} {'sure(s)':>8s}")
    for r in results:
        print(f"{r['cfg']['name']:14s} {r['final_train_loss']:8.4f} {r['final_val_loss']:8.4f} "
              f"{r['final_train_acc']:7.4f} {r['final_val_acc']:7.4f} "
              f"{r.get('test_acc', float('nan')):7.4f} {r['train_time_sec']:8.1f}")


if __name__ == "__main__":
    main()
