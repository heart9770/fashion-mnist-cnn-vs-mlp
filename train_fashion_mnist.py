"""Fashion-MNIST 服装图像分类实验：全连接网络 vs 卷积神经网络。

运行方式（工作区根目录，先设置 PYTHONPATH 指向 pylibs）：
    python train_fashion_mnist.py --out results --epochs 12 --seeds 42 43 44
数据从 data/fashion 下的 IDX 文件读取，模型与训练过程全部用 PyTorch 实现。
"""
import argparse
import gzip
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["T恤/上衣", "裤子", "套头衫", "连衣裙", "外套",
           "凉鞋", "衬衫", "运动鞋", "包", "短靴"]


# ---------------------------------------------------------------- 数据
def load_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        n = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        assert magic == 2051, magic
        buf = f.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, 1, rows, cols).copy()


def load_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        n = int.from_bytes(f.read(4), "big")
        assert magic == 2049, magic
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8).copy()


def load_dataset(root):
    xtr = load_idx_images(os.path.join(root, "train-images-idx3-ubyte.gz"))
    ytr = load_idx_labels(os.path.join(root, "train-labels-idx1-ubyte.gz"))
    xte = load_idx_images(os.path.join(root, "t10k-images-idx3-ubyte.gz"))
    yte = load_idx_labels(os.path.join(root, "t10k-labels-idx1-ubyte.gz"))
    return xtr, ytr, xte, yte


def to_float(batch_u8):
    """uint8 (B,1,28,28) -> float32 张量，缩放到 [0,1] 后按全局均值/标准差标准化。"""
    x = torch.from_numpy(batch_u8).float().div_(255.0)
    return (x - 0.2860) / 0.3530


# ---------------------------------------------------------------- 模型
class MLP(nn.Module):
    """784-256-128-10 三层全连接网络。"""

    def __init__(self, hidden=(256, 128), dropout=0.2):
        super().__init__()
        layers, prev = [], 28 * 28
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))


class SmallCNN(nn.Module):
    """两层卷积 + 两层全连接的轻量卷积网络。"""

    def __init__(self, use_bn=True, width=(16, 32), fc_dim=64, dropout=0.3):
        super().__init__()
        c1, c2 = width
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c1) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(c2) if use_bn else nn.Identity()
        self.fc1 = nn.Linear(c2 * 7 * 7, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 10)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = self.dropout(F.relu(self.fc1(x.flatten(1))))
        return self.fc2(x)


# ---------------------------------------------------------------- 数据增强
def augment(batch, pad=2):
    """随机平移（先补零再随机裁剪）+ 随机水平翻转。batch: (B,1,28,28)"""
    b = batch.shape[0]
    x = F.pad(batch, (pad, pad, pad, pad))
    size = 2 * pad + 1
    oy = torch.randint(0, size, (b,))
    ox = torch.randint(0, size, (b,))
    patches = x.unfold(2, 28, 1).unfold(3, 28, 1)  # (B,1,size,size,28,28)
    idx = torch.arange(b)
    out = patches[idx, :, oy, ox]
    flip = torch.rand(b) < 0.5
    if flip.any():
        out[flip] = torch.flip(out[flip], dims=[3])
    return out.contiguous()


# ---------------------------------------------------------------- 训练与评估
def iterate(x, y, batch_size, shuffle, seed=0):
    n = x.shape[0]
    order = np.random.RandomState(seed).permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, batch_size):
        sel = order[i:i + batch_size]
        yield x[sel], y[sel]


def predict(model, x_u8, batch_size=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, x_u8.shape[0], batch_size):
            out.append(model(to_float(x_u8[i:i + batch_size])).argmax(1).numpy())
    return np.concatenate(out)


def macro_f1(y_true, y_pred, n_class=10):
    f1s = []
    for c in range(n_class):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def train_one(name, model, xtr, ytr, xval, yval, epochs, batch_size, lr, use_aug,
              seed=42, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    hist = {"epoch": [], "train_loss": [], "train_acc": [], "val_acc": [], "epoch_time": []}
    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss, tot_correct, tot_n = 0.0, 0, 0
        for xb, yb in iterate(xtr, ytr, batch_size, shuffle=True, seed=seed * 1000 + ep):
            x = to_float(xb)
            if use_aug:
                x = augment(x)
            logits = model(x)
            loss = loss_fn(logits, torch.from_numpy(yb).long())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(yb)
            tot_correct += int((logits.argmax(1).detach().numpy() == yb).sum())
            tot_n += len(yb)
        sched.step()
        val_acc = float((predict(model, xval) == yval).mean())
        hist["epoch"].append(ep)
        hist["train_loss"].append(round(tot_loss / tot_n, 4))
        hist["train_acc"].append(round(tot_correct / tot_n, 4))
        hist["val_acc"].append(round(val_acc, 4))
        hist["epoch_time"].append(round(time.time() - t0, 2))
        log(f"[{name} s{seed}] ep {ep:02d}/{epochs} loss={tot_loss / tot_n:.4f} "
            f"train={tot_correct / tot_n:.4f} val={val_acc:.4f} ({time.time() - t0:.1f}s)")
    return {
        "name": name,
        "params": int(sum(p.numel() for p in model.parameters())),
        "train_time": round(time.time() - t_start, 1),
        "epoch_time_mean": round(float(np.mean(hist["epoch_time"])), 2),
        "history": hist,
    }


def summarize(result, model, xte, yte):
    pred = predict(model, xte)
    cm = np.zeros((10, 10), dtype=np.int64)
    for t, p in zip(yte, pred):
        cm[t, p] += 1
    result.update({
        "test_acc": round(float((pred == yte).mean()), 4),
        "macro_f1": round(macro_f1(yte, pred), 4),
        "per_class_acc": (cm.diagonal() / cm.sum(1)).round(4).tolist(),
        "confusion": cm.tolist(),
    })
    return pred


def build_model(kind):
    if kind == "MLP":
        return MLP()
    if kind == "CNN":
        return SmallCNN(use_bn=True)
    if kind == "CNN+Aug":
        return SmallCNN(use_bn=True)
    if kind == "CNN+Aug-NoBN":
        return SmallCNN(use_bn=False)
    raise ValueError(kind)


def run_config(kind, xtr, ytr, xval, yval, xte, yte, epochs, seed, log=print):
    # 关键：先固定种子再构造模型，否则权重初始化会用到进程启动时的随机状态，
    # 导致同一配置换个进程就复现不出同样的数值。
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(kind)
    use_aug = "Aug" in kind
    res = train_one(kind, model, xtr, ytr, xval, yval, epochs, 128, 1e-3, use_aug,
                    seed=seed, log=log)
    res["seed"] = seed
    pred = summarize(res, model, xte, yte)
    return res, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fashion")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--task", default="both", choices=["main", "small", "long", "both"])
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    xtr, ytr, xte, yte = load_dataset(args.data)
    xval, yval = xtr[54000:], ytr[54000:]
    xtr, ytr = xtr[:54000], ytr[:54000]
    print(f"train={xtr.shape} val={xval.shape} test={xte.shape}", flush=True)

    if args.task in ("main", "both"):
        runs = []
        for seed in args.seeds:
            for kind in ["MLP", "CNN", "CNN+Aug", "CNN+Aug-NoBN"]:
                res, pred = run_config(kind, xtr, ytr, xval, yval, xte, yte,
                                       args.epochs, seed)
                runs.append(res)
                if seed == args.seeds[0] and kind == "CNN+Aug":
                    np.savez(os.path.join(args.out, "preds_best.npz"), y_true=yte, y_pred=pred)
        agg = []
        for kind in ["MLP", "CNN", "CNN+Aug", "CNN+Aug-NoBN"]:
            sub = [r for r in runs if r["name"] == kind]
            accs = [r["test_acc"] for r in sub]
            f1s = [r["macro_f1"] for r in sub]
            agg.append({
                "name": kind,
                "params": sub[0]["params"],
                "test_acc_mean": round(float(np.mean(accs)), 4),
                "test_acc_std": round(float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0, 4),
                "macro_f1_mean": round(float(np.mean(f1s)), 4),
                "macro_f1_std": round(float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0, 4),
                "epoch_time_mean": sub[0]["epoch_time_mean"],
                "per_class_acc": sub[0]["per_class_acc"],
                "confusion": sub[0]["confusion"],
                "history": sub[0]["history"],
                "accs": accs,
            })
        payload = {"config": {"epochs": args.epochs, "batch_size": 128, "lr": 1e-3,
                              "seeds": args.seeds, "n_train": int(xtr.shape[0]),
                              "n_val": int(xval.shape[0]), "n_test": int(xte.shape[0])},
                   "models": agg, "runs": runs}
        with open(os.path.join(args.out, "main.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("MAIN", [(m["name"], m["test_acc_mean"], m["test_acc_std"]) for m in agg], flush=True)

    if args.task in ("small", "both"):
        # 小样本实验：每类 100 张，共 1000 张训练图片
        rs = np.random.RandomState(2024)
        idx = []
        for c in range(10):
            cls_idx = np.where(ytr == c)[0]
            idx.extend(rs.choice(cls_idx, 100, replace=False))
        idx = np.array(idx)
        rs.shuffle(idx)
        xs, ys = xtr[idx], ytr[idx]
        print(f"small-sample train={xs.shape}", flush=True)
        small = []
        for seed in args.seeds:
            for kind in ["MLP", "CNN", "CNN+Aug"]:
                res, _ = run_config(kind, xs, ys, xval, yval, xte, yte, 30, seed)
                small.append(res)
        agg_s = []
        for kind in ["MLP", "CNN", "CNN+Aug"]:
            sub = [r for r in small if r["name"] == kind]
            accs = [r["test_acc"] for r in sub]
            agg_s.append({"name": kind,
                          "test_acc_mean": round(float(np.mean(accs)), 4),
                          "test_acc_std": round(float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0, 4),
                          "macro_f1_mean": round(float(np.mean([r["macro_f1"] for r in sub])), 4),
                          "history": sub[0]["history"]})
        with open(os.path.join(args.out, "small.json"), "w", encoding="utf-8") as f:
            json.dump({"config": {"epochs": 30, "n_train": int(xs.shape[0]), "seeds": args.seeds},
                       "models": agg_s}, f, ensure_ascii=False, indent=2)
        print("SMALL", [(m["name"], m["test_acc_mean"]) for m in agg_s], flush=True)

    if args.task == "long":
        # 延长训练轮次：考察数据增强收益是否被收敛速度掩盖
        epochs = 30
        runs = []
        for seed in args.seeds:
            for kind in ["CNN", "CNN+Aug"]:
                res, _ = run_config(kind, xtr, ytr, xval, yval, xte, yte, epochs, seed)
                runs.append(res)
        agg_l = []
        for kind in ["CNN", "CNN+Aug"]:
            sub = [r for r in runs if r["name"] == kind]
            accs = [r["test_acc"] for r in sub]
            agg_l.append({"name": kind,
                          "test_acc_mean": round(float(np.mean(accs)), 4),
                          "test_acc_std": round(float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0, 4),
                          "macro_f1_mean": round(float(np.mean([r["macro_f1"] for r in sub])), 4),
                          "epoch_time_mean": sub[0]["epoch_time_mean"],
                          "train_time": sub[0]["train_time"],
                          "history": sub[0]["history"]})
        with open(os.path.join(args.out, "long.json"), "w", encoding="utf-8") as f:
            json.dump({"config": {"epochs": epochs, "n_train": int(xtr.shape[0]),
                                  "seeds": args.seeds}, "models": agg_l},
                      f, ensure_ascii=False, indent=2)
        print("LONG", [(m["name"], m["test_acc_mean"]) for m in agg_l], flush=True)


if __name__ == "__main__":
    main()