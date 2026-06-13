# SPDX-License-Identifier: MIT
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

DEFAULT_CPU_WORKERS = 128
INNER_WORKERS_PER_TASK = 1
AVAILABLE_GPUS = [0, 1, 2, 3]

CPU_THREAD_ENV: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "TORCH_NUM_THREADS": "1",
    "TORCH_NUM_INTEROP_THREADS": "1",
}


def resolve_cpu_workers(default: int = DEFAULT_CPU_WORKERS) -> int:
    """默认使用 128 个 CPU 并行；允许用 XRD_CPU_WORKERS 临时覆盖。"""
    raw = os.environ.get("XRD_CPU_WORKERS")
    if raw is None:
        return int(default)
    try:
        return max(1, int(raw))
    except ValueError:
        print(f"⚠️  XRD_CPU_WORKERS={raw!r} 无效，使用默认 {default}")
        return int(default)


def extract_wavelength(json_file: str) -> float:
    """从嵌套的 JSON 标签中提取主波长；解析失败时回退到 Cu Kα。"""
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        label_raw = data.get("label", "{}")
        label_data = json.loads(label_raw) if isinstance(label_raw, str) else label_raw

        xray_raw = label_data.get("xray_info", "{}") if isinstance(label_data, dict) else "{}"
        xray_info = json.loads(xray_raw) if isinstance(xray_raw, str) else xray_raw

        if isinstance(xray_info, dict):
            return float(xray_info.get("primary_wavelength", 1.5406))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return 1.5406


def refine_task(
    folder_path: str,
    folder_name: str,
    refinement_script: str,
    imp_dir: Optional[str],
    gpu_id: int,
) -> str:
    """多卡分流 + 输出落到对应 pattern 文件夹 + 记录最终耗时到 refine_log.txt。"""

    log_path = os.path.join(folder_path, "refine_log.txt")
    if os.path.exists(log_path):
        return f"跳过 {folder_name}: 已有精修记录"

    # pattern 目录内文件名（因为我们会 cwd=folder_path）
    json_file = os.path.join(folder_path, f"{folder_name}.json")
    xy_name = f"{folder_name}.xy"
    cif_name = f"{folder_name}.cif"

    # 仍然先在绝对路径上检查存在性（更可靠）
    xy_abs = os.path.join(folder_path, xy_name)
    cif_abs = os.path.join(folder_path, cif_name)
    if not (os.path.exists(xy_abs) and os.path.exists(cif_abs)):
        return f"跳过 {folder_name}: 缺少 .xy 或 .cif"

    wavelength = extract_wavelength(json_file)

    # 使用参数列表执行，保持 shell=False，避免命令注入面。
    cmd = [
        sys.executable,
        os.path.abspath(refinement_script),
        "--xy",
        f"./{xy_name}",
        "--main",
        f"./{cif_name}",
        "--wl",
        str(wavelength),
        "--num-workers",
        str(INNER_WORKERS_PER_TASK),
    ]
    if imp_dir:
        # imp_dir 也建议用绝对路径，避免 cwd 改变后找不到
        cmd.extend(["--imp", os.path.abspath(imp_dir)])

    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    my_env.update(CPU_THREAD_ENV)

    t0 = time.perf_counter()
    try:
        with open(log_path, "w", encoding="utf-8") as log_f:
            result = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                env=my_env,
                cwd=folder_path,
                check=False,
            )

        elapsed_s = time.perf_counter() - t0
        with open(log_path, "a", encoding="utf-8") as log_f:
            log_f.write("\n")
            log_f.write(f"⏱ Elapsed: {elapsed_s:.3f} s\n")
            log_f.write(f"🔚 ReturnCode: {result.returncode}\n")

        if result.returncode == 0:
            return f"成功: {folder_name} (GPU: {gpu_id}, {elapsed_s:.1f}s)"
        return f"失败: {folder_name} (GPU: {gpu_id}, {elapsed_s:.1f}s, 查看 refine_log.txt)"
    except (OSError, subprocess.SubprocessError) as e:
        elapsed_s = time.perf_counter() - t0
        try:
            with open(log_path, "a", encoding="utf-8") as log_f:
                log_f.write("\n")
                log_f.write(f"⏱ Elapsed: {elapsed_s:.3f} s\n")
                log_f.write(f"💥 Exception: {str(e)}\n")
        except OSError:
            pass
        return f"异常: {folder_name} -> {str(e)}"


def main() -> None:
    base = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(base, "database_opXRD")
    refine_script = os.path.join(base, "QL_yfs_XRD.py")
    impure_dir = os.path.join(base, "impure_phases")

    max_workers = resolve_cpu_workers()

    if not os.path.exists(data_root):
        print(f"错误: 找不到目录 {data_root}")
        return

    tasks_raw: list[tuple[str, str]] = []
    for item in sorted(os.listdir(data_root)):
        path = os.path.join(data_root, item)
        if os.path.isdir(path) and item.startswith("pattern_"):
            tasks_raw.append((path, item))

    print(f"🚀 准备4卡并行精修，共 {len(tasks_raw)} 个任务")
    print(f"⚙️  外层 CPU 并行总数: {max_workers} | 单任务内部 worker: {INNER_WORKERS_PER_TASK}")
    print(f"🎛️  GPU 分配方案: {AVAILABLE_GPUS}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_project = {}
        for i, (path, item) in enumerate(tasks_raw):
            gpu_id = AVAILABLE_GPUS[i % len(AVAILABLE_GPUS)]
            future = executor.submit(refine_task, path, item, refine_script, impure_dir, gpu_id)
            future_to_project[future] = item

        completed = 0
        for future in as_completed(future_to_project):
            item = future_to_project[future]
            try:
                res = future.result()
            except Exception as e:
                res = f"异常: {item} -> {str(e)}"
            completed += 1
            print(f"[{completed}/{len(tasks_raw)}] {res}")


if __name__ == "__main__":
    main()
