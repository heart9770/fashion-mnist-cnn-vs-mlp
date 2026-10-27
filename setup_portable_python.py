"""在项目目录里搭一套免安装的 Python 3.12 运行环境（不经过任何安装程序）。

产物：
    py312\\python.exe                      免安装 Python 3.12.10
    py312\\Lib\\site-packages\\...          torch / numpy / pillow 等库

用法：py312\\python.exe train_fashion_mnist.py ...
"""
import os
import shutil
import sys
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(BASE, "py312")
WHEELS = os.path.join(BASE, "wheels312")
SITE = os.path.join(PY, "Lib", "site-packages")

os.makedirs(PY, exist_ok=True)
os.makedirs(SITE, exist_ok=True)

# 1) 解压免安装 Python
embed = os.path.join(WHEELS, "python-embed.zip")
if not os.path.exists(os.path.join(PY, "python.exe")):
    with zipfile.ZipFile(embed) as zf:
        zf.extractall(PY)
    print("extracted portable python ->", PY)
else:
    print("portable python already present")

# 2) 让 site-packages 生效（官方 embed 包默认关闭 site）
pth = None
for name in os.listdir(PY):
    if name.endswith("._pth"):
        pth = os.path.join(PY, name)
        break
if pth is None:
    sys.exit("找不到 ._pth 文件")
with open(pth, "w", encoding="utf-8") as f:
    f.write("python312.zip\n.\nLib\\site-packages\n\nimport site\n")
print("patched", os.path.basename(pth))

# 3) 安装 wheel（直接解压 wheel，不调用 pip）
for name in sorted(os.listdir(WHEELS)):
    if not name.endswith(".whl"):
        continue
    with zipfile.ZipFile(os.path.join(WHEELS, name)) as zf:
        zf.extractall(SITE)
    print("installed", name)

print("done:", os.path.join(PY, "python.exe"))