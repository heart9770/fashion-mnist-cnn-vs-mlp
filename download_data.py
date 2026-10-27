"""下载 Fashion-MNIST 官方 IDX 数据到 data/fashion/。

用法：python download_data.py
数据来源：https://github.com/zalandoresearch/fashion-mnist
"""
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

os.makedirs("data/fashion", exist_ok=True)
for name in FILES:
    dest = os.path.join("data/fashion", name)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print("skip", name)
        continue
    print("downloading", name)
    urllib.request.urlretrieve(BASE + name, dest)
    print("  ->", os.path.getsize(dest), "bytes")
print("done")