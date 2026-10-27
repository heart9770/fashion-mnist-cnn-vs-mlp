"""把复跑结果与仓库中保存的结果逐项对比，判断复现是否成功。

用法：
    python verify_repro.py                 # 对比 results/ 与 results_check/ 的同一个随机种子
    python verify_repro.py --tol 0.2       # 允许 0.2 个百分点的差异（跨 PyTorch 版本时用）
    python verify_repro.py --new my_dir
"""
import argparse
import json
import os
import sys

MODELS = ["MLP", "CNN", "CNN+Aug", "CNN+Aug-NoBN"]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect(data, seed):
    """返回 {(name, seed): (test_acc, macro_f1)}"""
    out = {}
    for name, payload in (("runs", data.get("runs", [])), ("models", data.get("models", []))):
        for r in payload:
            if "seed" in r and r["seed"] == seed:
                out[(r["name"], seed)] = (r["test_acc"], r["macro_f1"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="results")
    ap.add_argument("--new", default="results_check")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tol", type=float, default=0.0,
                    help="允许的准确率差异（百分点），默认 0 表示要求完全一致")
    args = ap.parse_args()

    ref_path = os.path.join(args.ref, "main.json")
    new_path = os.path.join(args.new, "main.json")
    for p in (ref_path, new_path):
        if not os.path.exists(p):
            sys.exit(f"找不到 {p}，请先运行训练脚本")

    ref = collect(load(ref_path), args.seed)
    new = collect(load(new_path), args.seed)
    if not new:
        sys.exit(f"{new_path} 里没有随机种子 {args.seed} 的结果，请用 --seeds {args.seed} 运行")

    print(f"对比 {new_path} → {ref_path}（随机种子 {args.seed}，容差 {args.tol} 个百分点）\n")
    print(f"{'配置':<14}{'参考准确率':>12}{'本次准确率':>12}{'差值':>10}   结果")
    print("-" * 62)
    ok = True
    for name in MODELS:
        key = (name, args.seed)
        if key not in ref:
            print(f"{name:<14}{'—':>12}{'—':>12}{'—':>10}   参考文件缺该项")
            continue
        if key not in new:
            print(f"{name:<14}{'—':>12}{'—':>12}{'—':>10}   本次缺该项")
            ok = False
            continue
        r_acc, r_f1 = ref[key]
        n_acc, n_f1 = new[key]
        diff = (n_acc - r_acc) * 100
        passed = abs(diff) <= args.tol + 1e-9
        ok = ok and passed
        print(f"{name:<14}{r_acc * 100:>11.2f}%{n_acc * 100:>11.2f}%{diff:>+9.2f}pt   "
              f"{'一致' if passed else '不一致'}   (F1 {r_f1:.4f} → {n_f1:.4f})")

    print("-" * 62)
    if ok:
        print("结论：复现通过。同一环境下数值应完全一致；换 PyTorch 版本时允许 ±0.2pt 的数值差异。")
        return 0
    print("结论：存在超出容差的差异。请确认 epochs/batch_size/lr/seeds 与参考配置一致，"
          "并检查是否改动了数据划分。")
    return 1


if __name__ == "__main__":
    sys.exit(main())