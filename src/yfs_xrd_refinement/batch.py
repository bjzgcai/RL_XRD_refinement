# SPDX-License-Identifier: MIT
"""Resource-safe batch runner for opXRD-style pattern folders."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Sequence

INNER_WORKERS_PER_TASK = 1
CPU_THREAD_ENV: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "TORCH_NUM_THREADS": "1",
    "TORCH_NUM_INTEROP_THREADS": "1",
}


def resolve_gpu_ids(raw: Optional[str] = None) -> list[str]:
    """Resolve visible GPU labels without assuming a four-GPU machine."""
    value = raw or os.environ.get("XRD_GPU_IDS") or os.environ.get("CUDA_VISIBLE_DEVICES") or "0"
    gpu_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("at least one GPU id must be supplied")
    return gpu_ids


def resolve_outer_workers(gpu_ids: Sequence[str], requested: Optional[int] = None) -> int:
    """Default to one outer process per GPU; allow an explicit safe override."""
    if requested is None:
        raw = os.environ.get("XRD_OUTER_WORKERS")
        requested = int(raw) if raw is not None else len(gpu_ids)
    if int(requested) < 1:
        raise ValueError("max_workers must be at least 1")
    return int(requested)


def extract_wavelength(json_file: str) -> float:
    """Extract the primary wavelength; fall back to Cu K-alpha on bad metadata."""
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


def has_successful_output(folder_path: str) -> bool:
    """Only completed runs with final products are eligible for skipping."""
    folder = Path(folder_path)
    log_path = folder / "refine_log.txt"
    if not log_path.is_file():
        return False
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return_codes = re.findall(r"ReturnCode:\s*(-?\d+)", log_text)
    return_code_ok = bool(return_codes) and return_codes[-1] == "0"
    final_products = (folder / "yfsf_Refined.txt").is_file() and (folder / "yfsf_Refined.xy").is_file()
    return return_code_ok and final_products


def refine_task(
    folder_path: str,
    folder_name: str,
    refinement_script: Optional[str],
    imp_dir: Optional[str],
    gpu_id: str,
    inner_workers: int = INNER_WORKERS_PER_TASK,
) -> str:
    """Run one pattern with shell disabled and record wall time plus return code."""
    if has_successful_output(folder_path):
        return f"跳过 {folder_name}: 已有完整且成功的精修结果"

    folder = Path(folder_path)
    log_path = folder / "refine_log.txt"
    json_file = folder / f"{folder_name}.json"
    xy_name = f"{folder_name}.xy"
    cif_name = f"{folder_name}.cif"
    if not (folder / xy_name).is_file() or not (folder / cif_name).is_file():
        return f"跳过 {folder_name}: 缺少 .xy 或 .cif"

    wavelength = extract_wavelength(str(json_file))
    if refinement_script:
        command_prefix = [sys.executable, str(Path(refinement_script).resolve())]
    else:
        command_prefix = [sys.executable, "-m", "yfs_xrd_refinement.qlearning"]
    cmd = command_prefix + [
        "--xy", f"./{xy_name}",
        "--main", f"./{cif_name}",
        "--wl", str(wavelength),
        "--num-workers", str(max(1, int(inner_workers))),
    ]
    if imp_dir:
        cmd.extend(["--imp", str(Path(imp_dir).resolve())])

    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    child_env.update(CPU_THREAD_ENV)
    # Source checkouts need src/ on the child path; installed packages resolve
    # to site-packages here and remain importable as well.
    package_root = str(Path(__file__).resolve().parents[1])
    existing_pythonpath = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = (
        package_root if not existing_pythonpath
        else package_root + os.pathsep + existing_pythonpath
    )

    started = time.perf_counter()
    try:
        with log_path.open("w", encoding="utf-8") as log_f:
            result = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                env=child_env,
                cwd=str(folder),
                check=False,
            )
        elapsed_s = time.perf_counter() - started
        with log_path.open("a", encoding="utf-8") as log_f:
            log_f.write(f"\n⏱ Elapsed: {elapsed_s:.3f} s\n")
            log_f.write(f"🔚 ReturnCode: {result.returncode}\n")
        if result.returncode == 0 and has_successful_output(folder_path):
            return f"成功: {folder_name} (GPU: {gpu_id}, {elapsed_s:.1f}s)"
        if result.returncode == 0:
            return f"不完整: {folder_name} (返回码为 0，但缺少最终输出文件)"
        return f"失败: {folder_name} (GPU: {gpu_id}, {elapsed_s:.1f}s, 查看 refine_log.txt)"
    except (OSError, subprocess.SubprocessError) as exc:
        elapsed_s = time.perf_counter() - started
        try:
            with log_path.open("a", encoding="utf-8") as log_f:
                log_f.write(f"\n⏱ Elapsed: {elapsed_s:.3f} s\n")
                log_f.write(f"💥 Exception: {exc}\n")
        except OSError:
            pass
        return f"异常: {folder_name} -> {exc}"


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(repo_root / "database_opXRD"),
                        help="包含 pattern_* 子目录的数据根目录")
    parser.add_argument("--imp", default=None, help="候选杂相 CIF 目录；省略则只使用主相")
    parser.add_argument("--refiner", default=None,
                        help="可选精修入口脚本；默认使用 yfs_xrd_refinement.qlearning 模块")
    parser.add_argument("--gpus", default=None, help="逗号分隔的 GPU 编号")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="外层并发数；默认每张 GPU 一个任务")
    parser.add_argument("--inner-workers", type=int, default=INNER_WORKERS_PER_TASK,
                        help="每个精修任务生成 profile 的进程数")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    args = build_parser(repo_root).parse_args(argv)
    data_root = Path(args.data_root).resolve()
    refinement_script = args.refiner or None
    if refinement_script and not Path(refinement_script).is_file():
        raise FileNotFoundError(f"找不到精修入口: {refinement_script}")
    if args.imp and not Path(args.imp).is_dir():
        raise FileNotFoundError(f"找不到杂相目录: {args.imp}")
    if args.inner_workers < 1:
        raise ValueError("inner_workers must be at least 1")
    if not data_root.is_dir():
        raise FileNotFoundError(f"找不到数据目录: {data_root}")

    gpu_ids = resolve_gpu_ids(args.gpus)
    max_workers = resolve_outer_workers(gpu_ids, args.max_workers)
    tasks = [
        (str(path), path.name)
        for path in sorted(data_root.iterdir())
        if path.is_dir() and path.name.startswith("pattern_")
    ]
    max_workers = min(max_workers, max(1, len(tasks)))

    print(f"🚀 准备精修，共 {len(tasks)} 个任务")
    print(f"⚙️  外层并发: {max_workers} | 单任务内部 worker: {args.inner_workers}")
    print(f"🎛️  GPU 分配方案: {gpu_ids}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for index, (folder_path, folder_name) in enumerate(tasks):
            gpu_id = gpu_ids[index % len(gpu_ids)]
            future = executor.submit(
                refine_task,
                folder_path,
                folder_name,
                refinement_script,
                args.imp,
                gpu_id,
                args.inner_workers,
            )
            futures[future] = folder_name

        for completed, future in enumerate(as_completed(futures), start=1):
            folder_name = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = f"异常: {folder_name} -> {exc}"
            print(f"[{completed}/{len(tasks)}] {result}")


if __name__ == "__main__":
    main()
