# 复现与验证指南

本文件说明如何在一台普通电脑上复现报告中的全部实验，并判断复现是否成功。
文中所有数字都是本机实际跑出来的，可以直接对照。

> **装不上 Python？（联想电脑管家拦截安装包、点“忽略”后窗口不出现）**
> 项目里已经配好一套**免安装的 Python 3.12 + PyTorch**（`py312\` 文件夹），
> 不经过任何安装程序。直接双击 **`run_verify.bat`** 即可完成训练与比对，约 5 分钟。
> 本机实测四组配置与 `results/` 完全一致（MLP 89.18%、CNN 91.72%、CNN+Aug 90.03%、
> CNN+Aug-NoBN 88.99%）。下面的“快速验证”“完整复现”两节是在已装好 Python 3.12 的前提下写的。

> **第一次操作、不熟悉命令行？** 请先看 [手动复现手册.md](手动复现手册.md)：里面有逐条命令、
> 每步应有的画面和常见报错处理；也可以直接双击 setup_and_verify.bat 一键完成。
> 本机已放好 64 位 Python 3.12 安装包（python-3.12.10-amd64.exe），当前系统里的 Python 是
> 3.8 32 位，装不了 PyTorch，必须先升到 3.12。

---

## 一、要复现的东西

报告里的数字来自三组实验，全部由本目录的脚本生成：

| 报告位置 | 实验 | 规模 | 耗时（16 核 CPU） |
| --- | --- | --- | --- |
| 表 2 | 主实验 | 4 组配置 × 3 个随机种子 × 12 轮 | 约 15 分钟 |
| 表 3 | 小样本实验 | 3 组配置 × 3 个种子 × 30 轮（1000 张） | 约 3 分钟 |
| 5.4 节 | 延长训练对照 | 2 组配置 × 2 个种子 × 30 轮 | 约 16 分钟 |

判定标准：同一软件环境、同一线程数下，`test_acc` 应与 `results/` 里的完全一致；
换了 PyTorch 版本时允许 ±0.2 个百分点的数值差异（原因见第八节）。

---

## 二、准备环境

### 1. 确认 Python

```bash
python --version
```

需要 3.10 ~ 3.12。Windows 上如果提示“不是内部或外部命令”，说明没装 Python 或没加 PATH。

### 2. 安装依赖（CPU 版 PyTorch，不需要显卡）

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`requirements.txt` 里除 torch 外只有 numpy、pillow（绘图）和 python-docx（重新生成 Word）。
下载慢时把第一条换成清华镜像：

```bash
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 自检

```bash
python -c "import torch, numpy, PIL; print(torch.__version__, torch.get_num_threads())"
```

输出形如 `2.14.1+cpu 8` 即可（版本号不要求与本文一致，线程数取决于你的 CPU）。

---

## 三、下载数据

```bash
python download_data.py
```

运行后 `data/fashion/` 下会出现 4 个 `.gz`，共约 30 MB，来自 Fashion-MNIST 官方 GitHub 仓库。

如果访问 GitHub 有困难，可手动下载这四个文件放到 `data/fashion/`：

```
train-images-idx3-ubyte.gz    train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz     t10k-labels-idx1-ubyte.gz
```

地址前缀：`https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/`

---

## 四、快速验证（推荐先做这一步）

### 1. 只跑一个随机种子

```bash
python train_fashion_mnist.py --out results_check --epochs 12 --seeds 42 --task main
```

约 5 分钟。`--task main` 表示只跑主实验（4 组配置）；去掉它会顺带跑小样本实验，多花约 3 分钟。
控制台逐轮打印 `loss / train_acc / val_acc`，最后一行打印四组配置的测试准确率。

Windows 用户也可以直接双击 `setup_and_verify.bat`：它会检查 Python、建虚拟环境、装依赖、下数据、跑 1 个种子并自动比对，全程不用输入命令。

### 2. 一键比对

```bash
python verify_repro.py --new results_check --tol 0
```

本机实测输出（2026-10 实测，四行全部一致）：

```
对比 results_check\main.json → results\main.json（随机种子 42，容差 0.0 个百分点）

配置                 参考准确率      本次准确率       差值   结果
--------------------------------------------------------------
MLP                 89.18%      89.18%    +0.00pt   一致   (F1 0.8917 → 0.8917)
CNN                 91.72%      91.72%    +0.00pt   一致   (F1 0.9171 → 0.9171)
CNN+Aug             90.03%      90.03%    +0.00pt   一致   (F1 0.9002 → 0.9002)
CNN+Aug-NoBN        88.99%      88.99%    +0.00pt   一致   (F1 0.8894 → 0.8894)
--------------------------------------------------------------
结论：复现通过。
```

四行都显示“一致”即复现成功。任何一行出现偏差，按第七节排查。

---

## 五、完整复现（重跑报告里所有实验）

```bash
:: 1) 备份原有结果，避免被覆盖
move results results_origin

:: 2) 主实验 + 小样本实验（3 个种子）
python train_fashion_mnist.py --out results --epochs 12 --seeds 42 43 44

:: 3) 延长训练对照（30 轮）
python train_fashion_mnist.py --out results --task long --seeds 42 43

:: 4) 重新生成插图与 Word 报告
python make_figures.py
python build_report.py
```

说明：

- 第 2 步不加 `--task` 时默认为 `both`，即主实验与小样本实验一起跑。
- 第 4 步 `build_report.py` 读取 `results/*.json` 重新生成
  `深度学习课程设计技术报告.docx`，所以报告里的表格与正文数字永远和你新跑出来的结果一致。
- `make_figures.py` 只依赖 `results/*.json` 和 `data/fashion/`，不重新训练。

---

## 六、对照表：应当得到的数字

### 6.1 随机种子 42（单次运行，用于快速验证）

| 配置 | 测试准确率 | 宏平均 F1 | 参数量 |
| --- | --- | --- | --- |
| MLP | 89.18% | 0.8917 | 235,146 |
| CNN | 91.72% | 0.9171 | 105,962 |
| CNN+Aug | 90.03% | 0.9002 | 105,962 |
| CNN+Aug-NoBN | 88.99% | 0.8894 | 105,866 |

### 6.2 三个种子 42/43/44 的均值±标准差（报告表 2）

| 配置 | 测试准确率 | 宏平均 F1 | 单轮耗时/s |
| --- | --- | --- | --- |
| MLP | 89.29±0.10 | 0.8927 | 1.6 |
| CNN | 91.62±0.10 | 0.9160 | 8.9 |
| CNN+Aug | 90.09±0.05 | 0.9005 | 8.2 |
| CNN+Aug-NoBN | 89.15±0.15 | 0.8907 | 6.5 |

### 6.3 另外两组

| 实验 | 结果 |
| --- | --- |
| 小样本（每类 100 张，30 轮） | MLP 81.23%、CNN 83.71%、CNN+Aug 80.91% |
| 延长训练（30 轮，种子 42/43） | CNN 92.08%、CNN+Aug 91.39% |

---

## 七、结果对不上怎么办

| 现象 | 可能原因 | 处理 |
| --- | --- | --- |
| 差距 ≤ 0.3 个百分点 | PyTorch 版本不同、CPU 多线程归约顺序不同 | 用 `--tol 0.3` 再比对 |
| 某组配置差距很大 | 改了 `--epochs`、学习率、批大小 | 与第五节的命令逐字对照 |
| 准确率只有 10% 左右 | 数据文件损坏或放错目录 | 删掉 `data/fashion/` 后重新 `python download_data.py` |
| 报错 `ModuleNotFoundError: torch` | 依赖装到了另一个 Python 解释器 | 用 `python -m pip install ...` 安装，确保与运行脚本是同一个解释器 |
| `找不到 results_check\main.json` | 上一步训练没跑完或输出目录名不对 | 确认 `--out results_check`，再跑比对 |
| 想排除线程影响 | 多线程求和顺序不确定 | `set OMP_NUM_THREADS=1` 后重跑（可能与报告差 ±0.1pt） |

---

## 八、关于“能不能复现”的诚实说明

1. **已修复的缺陷**：早期版本在构造模型前没有调用 `torch.manual_seed(seed)`，导致每个进程里
   第一个模型（MLP）的权重初始化取到的是进程启动时的随机状态，换个进程重跑会差 0.08 个百分点
   （其余三组因为训练函数内部已经设过种子，所以完全一致）。现在 `run_config()` 在
   `build_model()` 之前就固定了 `torch` 与 `numpy` 的种子，四组配置都能逐位复现，
   上面的实测输出就是修复后重新跑出来的。
2. **仍然存在的非确定性**：CPU 多线程求和的归约顺序、不同 PyTorch 版本的算子实现，
   通常带来 0.0~0.2 个百分点的波动。要彻底消除可设 `OMP_NUM_THREADS=1`。
3. **结论**：同一台机器、同一版本、同样线程数 = 完全一致；换环境 = 允许 ±0.2 个百分点。
   这也是报告 2.2 节写“固定随机种子”的依据。

---

## 九、只检查结果文件、不训练

```bash
python -c "import json;d=json.load(open('results/main.json',encoding='utf-8'));print([(m['name'],m['test_acc_mean']) for m in d['models']])"
```

应打印：

```
[('MLP', 0.8929), ('CNN', 0.9162), ('CNN+Aug', 0.9009), ('CNN+Aug-NoBN', 0.8915)]
```

可用于确认 `results/` 没有被改动过。