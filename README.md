# Fashion-MNIST 服装图像分类：CNN 与全连接网络的对比实验

深度学习课程设计报告的配套代码。在同一数据集、同一训练预算下比较
全连接网络（MLP）与轻量卷积网络（SmallCNN），并对数据增强、批量归一化（BN）做消融，
另含小样本实验与延长训练轮次的对照实验。

## 环境

- Python 3.10+
- PyTorch 2.x（CPU 版本即可，无需 GPU）
- NumPy、Pillow；生成 Word 报告另需 python-docx

```bash
pip install -r requirements.txt
```

实验在 Windows 11 / 16 逻辑核 CPU / Python 3.12 / PyTorch 2.14.1 上完成，
全部训练 12 轮时耗时约 15 分钟（3 个随机种子 × 4 组配置）。

## 数据

Fashion-MNIST（Xiao 等，2017），70000 张 28×28 灰度图像，10 类。
本项目不转载数据，运行下面命令从官方仓库下载（约 30 MB）：

```bash
python download_data.py
```

数据会保存到 `data/fashion/`，共 4 个 IDX 文件。划分方式：官方训练集前 54000 张用于训练、
后 6000 张作为验证集，官方测试集 10000 张仅用于最终评估。

## 运行

```bash
# 1. 主实验（4 组配置）+ 小样本实验，3 个随机种子
python train_fashion_mnist.py --out results --epochs 12 --seeds 42 43 44

# 2. 延长到 30 轮的对照实验（数据增强的收敛性讨论）
python train_fashion_mnist.py --out results --task long --seeds 42 43

# 3. 生成插图
python make_figures.py

# 4.（可选）重新生成 Word 报告，需 python-docx
python build_report.py
```

结果写入 `results/*.json`，插图写入 `figs/`。

## 主要结果（12 轮，3 个种子的均值±标准差）

| 配置 | 测试准确率/% | 宏平均 F1 | 参数量 | 单轮耗时/s |
| --- | --- | --- | --- | --- |
| MLP | 89.29±0.10 | 0.8927 | 235,146 | 1.6 |
| CNN | 91.62±0.10 | 0.9160 | 105,962 | 8.9 |
| CNN+Aug | 90.09±0.05 | 0.9005 | 105,962 | 8.2 |
| CNN+Aug-NoBN | 89.15±0.15 | 0.8907 | 105,866 | 6.5 |

小样本（每类 100 张，共 1000 张，30 轮）：MLP 81.23%、CNN 83.71%、CNN+Aug 80.91%。
延长到 30 轮后 CNN 92.08%、CNN+Aug 91.39%。

## 文件说明

| 文件 | 功能 |
| --- | --- |
| `download_data.py` | 下载 Fashion-MNIST 官方 IDX 数据 |
| `train_fashion_mnist.py` | 数据解析、MLP/SmallCNN 定义、数据增强、训练与评估主流程 |
| `verify_repro.py` | 把复跑结果与 `results/` 逐项比对，判断复现是否成功 |
| `make_figures.py` | 由结果文件生成样例图、训练曲线、混淆矩阵、结构图 |
| `plot_utils.py` | 基于 Pillow 的轻量绘图工具 |
| `build_report.py` | 依据结果文件生成技术报告 Word 文档 |
| `run_verify.bat` | 免安装一键验证：直接用 `py312\` 里的 Python 跑 1 个种子并与 `results/` 比对 |
| `setup_and_verify.bat` | 正规安装路线的一键脚本：检查 Python 3.12–3.14 → 建 venv → 装依赖 → 下数据 → 训练 → 比对 |
| `setup_portable_python.py` | 用官方免安装包在 `py312\` 建一套 Python 3.12 + torch 环境（不需要任何安装程序） |
| `REPRODUCE.md` | 复现与验证的完整步骤、对照数字与排查表 |
| `手动复现手册.md` | 面向不熟悉命令行的逐步操作说明（含每步预期输出） |
| `python-3.12.10-amd64.exe` | 64 位 Python 3.12 安装包（本机原为 3.8 32 位，无法装 PyTorch） |
| `results/` | 训练结果（JSON）与预测结果 |
| `figs/` | 报告中使用的插图 |

## 复现验证

**方式一（免安装，双击即可）**：双击 `run_verify.bat`。它用项目内 `py312\` 里的
Python 3.12 + torch 跑一个种子并自动比对，全程不经过安装程序，也不需要联网装依赖。

**方式二（已装好 Python 3.12 的环境）**：

```bash
python train_fashion_mnist.py --out results_check --epochs 12 --seeds 42
python verify_repro.py --new results_check --tol 0
```

四行都是“一致”说明复现成功。本机实测（种子 42）：
MLP 89.18%、CNN 91.72%、CNN+Aug 90.03%、CNN+Aug-NoBN 88.99%，与 `results/` 完全一致。
更完整的步骤见 [REPRODUCE.md](REPRODUCE.md) 与 [手动复现手册.md](手动复现手册.md)。

## 说明

- 数据增强用 `unfold` 构造候选窗口，一次性完成整批随机裁剪，避免逐样本循环。
- BN 消融由 `SmallCNN(use_bn=False)` 控制，其余结构不变。
- 每个配置在**构造模型之前**就固定 `torch` 与 `numpy` 的随机种子，保证换进程也能逐位复现。
- 图像与结构图均由脚本生成，未手工修改。

## 参考

1. Xiao H, Rasul K, Vollgraf R. Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv:1708.07747, 2017.
2. Ioffe S, Szegedy C. Batch Normalization. ICML, 2015.
3. Kingma D P, Ba J. Adam: A Method for Stochastic Optimization. ICLR, 2015.
