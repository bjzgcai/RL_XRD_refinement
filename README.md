# yfs_XRD_refinement

版本：v1

本项目用于对 XRD 谱图进行自动化精修。当前主要入口有两个：

- `yfs_XRD.py`：常规多阶段自动精修版本。
- `QL_yfs_XRD.py`：在常规流程基础上加入 Q-Learning 动作选择的版本，适合尝试更智能的参数搜索。

两个脚本都会读取实验谱 `.xy` 文件、主相 CIF 文件和可选杂相 CIF 目录，自动进行候选杂相筛选、相组合搜索、峰形/晶胞/原子位置/占位/择优取向等参数优化，并输出拟合图、拟合曲线、精修报告和精修后的 CIF。

## 环境安装

建议先创建独立 Python 环境，然后安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：

- `numpy`
- `scipy`
- `torch`
- `matplotlib`
- `pymatgen`

程序会自动检测是否可用 CUDA；如果可用，会优先使用 GPU，否则使用 CPU。

## 输入文件准备

推荐在一个样品目录中放置以下文件：

```text
sample_dir/
  sample.xy
  main.cif
  impure_phase/
    impurity_1.cif
    impurity_2.cif
```

说明：

- `.xy` 文件：实验 XRD 谱，至少两列，第一列为 `2Theta`，第二列为强度。
- 主相 CIF：用 `--main` 指定，例如 `main.cif`。
- 杂相目录：用 `--imp` 指定，例如 `impure_phase`。目录下所有 `.cif` 会作为候选杂相参与筛选。
- 如果不指定 `--xy`，程序会自动选择当前目录下按文件名排序的第一个 `.xy` 文件。

## 基本用法

进入样品目录后运行：

```bash
python yfs_XRD.py --xy sample.xy --main main.cif --imp impure_phase
```

使用 Q-Learning 版本：

```bash
python QL_yfs_XRD.py --xy sample.xy --main main.cif --imp impure_phase
```

如果只做单相精修，可以不提供杂相目录：

```bash
python yfs_XRD.py --xy sample.xy --main main.cif
```

## 常用参数

两个脚本都支持：

```bash
--xy              实验谱文件，格式为 .xy
--main            主相 CIF 文件
--imp             杂相 CIF 文件夹
--num-workers     并行进程数，默认使用 CPU 核心数
--main-bias       主相偏置系数，只影响相组合筛选阶段，默认 1.0
--stoich-phase    化学计量约束参考相，通常与主相相同
--stoich          目标化学计量比，例如 "Li:6,S:5,P:1,Cl:1"
--lambda-stoich   化学计量约束强度，默认 0.5
```

`QL_yfs_XRD.py` 额外支持：

```bash
--wl              X-ray 波长，默认 1.5406 Angstrom
```

示例：

```bash
python QL_yfs_XRD.py \
  --xy sample.xy \
  --main main.cif \
  --imp impure_phase \
  --wl 1.5406 \
  --num-workers 8 \
  --stoich "Li:6,S:5,P:1,Cl:1" \
  --lambda-stoich 0.5
```

## 输出文件

运行完成后，常见输出包括：

```text
yfsf_Refined.png          最终拟合图
yfsf_Refined.xy           实验谱与拟合谱数据
yfsf_Refined.txt          最终报告，包含 Rwp、相分数、scale、TCH 参数等
yfsf_refined_cifs/        精修后的 CIF 文件
refine_log.csv            精修过程日志
Rwp_curve.png             Rwp 变化曲线
phase_fraction_curve.png  相分数变化曲线
scale_curve.png           scale 变化曲线
```

`yfs_XRD.py` 还会保存阶段结果，例如：

```text
stage1_output/
stage2_output/
```

这些目录中包含阶段拟合图、阶段拟合曲线、阶段报告和阶段 CIF。

## 两个脚本如何选择

优先建议先运行：

```bash
python yfs_XRD.py --xy sample.xy --main main.cif --imp impure_phase
```

如果常规版本陷入局部最优、某些参数调整效果不稳定，或希望尝试强化学习动作选择，再运行：

```bash
python QL_yfs_XRD.py --xy sample.xy --main main.cif --imp impure_phase
```

`QL_yfs_XRD.py` 的搜索过程通常更复杂，运行时间可能更长。

## 精修效果不佳时的建议

如果最终 Rwp 偏高、背景拟合异常、低角度背景过强，或拟合曲线明显被背景拖偏，可以先手动扣除背景，再用扣背景后的 `.xy` 文件重新运行精修。

推荐流程：

1. 用常用 XRD 软件或自己的脚本对原始谱图进行背景扣除。
2. 导出新的两列 `.xy` 文件，仍保持第一列为 `2Theta`，第二列为扣背景后的强度。
3. 用新的 `.xy` 文件重新运行：

```bash
python yfs_XRD.py --xy sample_bg_removed.xy --main main.cif --imp impure_phase
```

如果扣背景后仍然效果不佳，可以继续检查主相 CIF 是否正确、候选杂相是否完整、波长是否匹配，以及是否存在明显择优取向或峰位系统偏移。

## 注意事项

- `.xy` 强度会在程序内部归一化。
- 候选杂相越多，组合搜索和精修时间越长。
- `--num-workers` 不宜超过机器可承受范围，过大可能导致内存压力。
- `--stoich` 格式必须为英文冒号和英文逗号，例如 `"Li:6,S:5,P:1,Cl:1"`。
- 如果 CIF 中缺少 `Uiso`，程序会使用默认值并在导出时尽量补齐。
