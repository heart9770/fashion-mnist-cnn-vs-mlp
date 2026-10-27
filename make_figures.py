"""根据实验结果生成报告插图：数据样例、训练曲线、混淆矩阵、网络结构图。

图中文字按最终排版宽度（约 10–13 cm）设定相对字号，保证打印后可读。
"""
import json
import os

import numpy as np
from PIL import Image

import plot_utils as pu
from train_fashion_mnist import CLASSES, load_dataset

RESULTS = "results"
FIGS = "figs"
BLUE = (43, 108, 176)
ORANGE = (221, 107, 32)
GREEN = (47, 133, 90)
RED = (197, 48, 48)
SHOW_TITLE = False  # 图中不画标题，由 Word 图题承担
SHORT = ["T恤", "裤子", "套头衫", "连衣裙", "外套", "凉鞋", "衬衫", "运动鞋", "包", "短靴"]

os.makedirs(FIGS, exist_ok=True)


def upscale(img_u8, scale=3):
    im = Image.fromarray(img_u8.reshape(28, 28)).convert("L")
    return im.resize((28 * scale, 28 * scale), Image.NEAREST).convert("RGB")


def fig_samples(xte, yte, scale=2, cols=10, rows=2, label_size=17):
    cell = 28 * scale + 10
    left, top, bottom = 26, (56 if SHOW_TITLE else 6), 30
    w = left + cols * cell + 16
    h = top + rows * cell + bottom
    img, d = pu.canvas(w, h)
    for c in range(cols):
        idx = np.where(yte == c)[0][:rows]
        for r, i in enumerate(idx):
            im = upscale(xte[i], scale)
            x = left + c * cell
            y = top + r * cell
            img.paste(im, (x, y))
            d.rectangle([x - 1, y - 1, x + 28 * scale, y + 28 * scale], outline=(205, 205, 205))
        pu.text(d, (left + c * cell + 14 * scale, top + rows * cell + 3), SHORT[c],
                size=label_size, anchor="ma")
    return pu.save(img, os.path.join(FIGS, "fig1_samples.png"))


def fig_curves(main, w=760, h=330, tick_size=18, label_size=19):
    models = main["models"]
    sel = [m for m in models if m["name"] in ("MLP", "CNN", "CNN+Aug")]
    colors = {"MLP": ORANGE, "CNN": BLUE, "CNN+Aug": GREEN}
    img, d = pu.canvas(w, h)
    panels = [
        ("(a) 训练损失", "train_loss", (0.0, 1.0), 0.2, "{:.1f}"),
        ("(b) 验证集准确率", "val_acc", (0.80, 0.94), 0.02, "{:.2f}"),
    ]
    epochs = [e for e in sel[0]["history"]["epoch"]]
    for k, (title, key, ylim, ystep, fmt) in enumerate(panels):
        left = 92 + k * 372
        rect = (left, 52, left + 250, 252)
        ylo, yhi = ylim
        nticks = int(round((yhi - ylo) / ystep))
        yticks = [round(ylo + i * ystep, 3) for i in range(nticks + 1)]
        fx, fy = pu.axes(d, rect, (1, len(epochs)), ylim,
                         xticks=list(range(1, len(epochs) + 1, max(1, len(epochs) // 5))),
                         yticks=yticks, xlabel="训练轮次 (epoch)", ylabel=title,
                         y_fmt=fmt, tick_size=tick_size, label_size=label_size,
                         x_fmt="{:.0f}")
        for m in sel:
            vals = m["history"][key]
            pts = [(fx(e), fy(v)) for e, v in zip(epochs, vals)]
            pu.polyline(d, pts, colors[m["name"]], width=3)
            pu.markers(d, pts, colors[m["name"]], r=3)
    pu.legend(d, [(m["name"], colors[m["name"]]) for m in sel], (w - 168, 14), size=17)
    return pu.save(img, os.path.join(FIGS, "fig2_curves.png"))


def fig_confusion(main, cell=36, label_size=16):
    best = max(main["models"], key=lambda m: m["test_acc_mean"])
    cm = np.array(best["confusion"])
    left, top = 132, 14
    w = left + 10 * cell + 26
    h = top + 10 * cell + 92
    img, d = pu.canvas(w, h)
    vmax = cm.max()
    for i in range(10):
        for j in range(10):
            v = cm[i, j] / vmax
            c = (int(255 - 205 * v), int(250 - 165 * v), int(255 - 80 * v))
            x, y = left + j * cell, top + i * cell
            d.rectangle([x, y, x + cell, y + cell], fill=c, outline=(255, 255, 255), width=2)
            if cm[i, j] > 0:
                tc = (255, 255, 255) if v > 0.55 else (60, 60, 60)
                pu.text(d, (x + cell / 2, y + cell / 2), str(cm[i, j]), size=label_size - 1,
                        fill=tc, anchor="mm")
    for i in range(10):
        pu.text(d, (left - 6, top + i * cell + cell / 2), SHORT[i], size=label_size, anchor="rm")
        pu.text(d, (left + i * cell + cell / 2, top + 10 * cell + 26), SHORT[i],
                size=label_size, anchor="ma")
    pu.text(d, (left + 5 * cell, top + 10 * cell + 56), "预测类别", size=label_size + 1, anchor="ma")
    for k, ch in enumerate("真实类别"):
        pu.text(d, (20, top + 5 * cell - 56 + k * 21), ch, size=label_size, anchor="ma")
    return pu.save(img, os.path.join(FIGS, "fig3_confusion.png"))


def fig_arch(w=1400, h=430):
    img, d = pu.canvas(w, h)
    blocks = [
        ("输入\n1×28×28", (60, 90, 120)),
        ("卷积块 1\n3×3 Conv, 16\nBN+ReLU+池化", (232, 240, 250)),
        ("卷积块 2\n3×3 Conv, 32\nBN+ReLU+池化", (232, 240, 250)),
        ("分类头\nFC 64 + Dropout\nFC 10", (232, 240, 250)),
        ("输出\n10 类概率", (60, 90, 120)),
    ]
    top, bh = 96, 210
    margin, gap = 40, 34
    bw = (w - 2 * margin - gap * (len(blocks) - 1)) / len(blocks)
    for i, (label, _c) in enumerate(blocks):
        x = margin + i * (bw + gap)
        fill = (232, 240, 250) if 0 < i < len(blocks) - 1 else (243, 243, 243)
        d.rounded_rectangle([x, top, x + bw, top + bh], radius=14, fill=fill,
                            outline=(120, 145, 175), width=3)
        lines = label.split("\n")
        for j, ln in enumerate(lines):
            pu.text(d, (x + bw / 2, top + bh / 2 + (j - (len(lines) - 1) / 2) * 46), ln,
                    size=30, anchor="mm")
        if i < len(blocks) - 1:
            y = top + bh / 2
            x2 = x + bw + gap
            d.line([x + bw + 4, y, x2 - 18, y], fill=(110, 110, 110), width=4)
            d.polygon([(x2 - 18, y - 11), (x2 - 18, y + 11), (x2 - 2, y)], fill=(110, 110, 110))
    pu.text(d, (w / 2, 34), "每两个卷积块之后做一次 2×2 最大池化：28×28 → 14×14 → 7×7",
            size=26, anchor="ma", fill=(90, 90, 90))
    return pu.save(img, os.path.join(FIGS, "fig5_arch.png"))


def main():
    with open(os.path.join(RESULTS, "main.json"), encoding="utf-8") as f:
        main_json = json.load(f)
    _, _, xte, yte = load_dataset("data/fashion")
    print(fig_samples(xte, yte))
    print(fig_curves(main_json))
    print(fig_confusion(main_json))
    print(fig_arch())


if __name__ == "__main__":
    main()