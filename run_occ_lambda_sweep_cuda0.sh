#!/usr/bin/env bash
set -Eeuo pipefail

# Run the four mixed-occupancy examples on physical CUDA GPU 0 for
# lambda_stoich = 0.3, 1.0, and 2.0. Jobs run sequentially by default so one
# GPU is not oversubscribed. Every invocation uses a fresh output root and
# copies its CIF/XY inputs into an isolated task directory.
#
# Usage:
#   bash run_occ_lambda_sweep_cuda0.sh
#
# Rerun only the two corrected Na mappings:
#   SAMPLE_FILTER="Na0.67Ni0.33Mn0.67O2,Na0.58Ni0.33Mn0.67O1.95" \
#     bash run_occ_lambda_sweep_cuda0.sh
#
# Optional environment overrides:
#   NUM_WORKERS=8                 CPU workers used for profile generation
#   STAGE_LOOPS="60 100 150"      Q-learning loops for stages 1, 2, and 3
#   SEED=20260820                 common seed used by every comparison
#   WAVELENGTH=1.5406             Cu-target wavelength in Angstrom
#   MAX_FORMULA_DEVIATION=0.02    input/target formula-unit tolerance
#   PYTHON_BIN=/path/to/python    override the default XRD Conda Python
#   OUTPUT_ROOT=/new/results      must not already exist
#   SAMPLE_FILTER="name1,name2"   optional logical-sample filter
#   DRY_RUN=1                     validate inputs and print commands only

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OCC_DIR="${PROJECT_DIR}/Occ refinement"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OCC_DIR}/lambda_stoich_sweep_cuda0_corrected_${RUN_TAG}}"
PYTHON_BIN="${PYTHON_BIN:-/vepfs-mlp2/project-battery/yinfusheng/myenvs_dir/XRD/bin/python}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-20260820}"
STAGE_LOOPS="${STAGE_LOOPS:-60 100 150}"
WAVELENGTH="${WAVELENGTH:-1.5406}"
MAX_FORMULA_DEVIATION="${MAX_FORMULA_DEVIATION:-0.02}"
SAMPLE_FILTER="${SAMPLE_FILTER:-}"
DRY_RUN="${DRY_RUN:-0}"

LAMBDAS=("0.3" "1.0" "2.0")

# Verified logical-sample -> physical-directory mapping. The two Na directory
# names are reversed relative to the structural formulas inside their CIFs.
SAMPLE_NAMES=(
  "Li2HfCl6"
  "LiMn1.5Ni0.5O4"
  "Na0.67Ni0.33Mn0.67O2"
  "Na0.58Ni0.33Mn0.67O1.95"
)
SOURCE_DIRS=(
  "Li2HfCl6"
  "LiMn1.5Ni0.5O4"
  "Na0.56Ni0.333Mn0.667O1.95"
  "Na0.67Ni0.33Mn0.67O2"
)
XY_FILES=(
  "Li2HfCl6.xy"
  "Refined.xy"
  "Refined.xy"
  "Refined.xy"
)
STOICH_TARGETS=(
  "Li:2,Hf:1,Cl:6"
  "Li:1,Mn:1.5,Ni:0.5,O:4"
  "Na:0.67,Ni:0.33,Mn:0.67,O:2"
  "Na:0.58,Ni:0.33,Mn:0.67,O:1.95"
)
ANIONS=("Cl" "O" "O" "O")
IDENTITY_MARKERS=(
  "_space_group_IT_number                 62"
  "_database_code_ICSD 188648"
  "_database_code_ICSD 192731"
  "_database_code_ICSD 90113"
)

read -r -a STAGE_LOOP_ARGS <<< "${STAGE_LOOPS}"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

sample_is_selected() {
  local candidate="$1"
  [[ -z "${SAMPLE_FILTER}" || ",${SAMPLE_FILTER}," == *",${candidate},"* ]]
}

[[ "${NUM_WORKERS}" =~ ^[1-9][0-9]*$ ]] \
  || die "NUM_WORKERS must be a positive integer (received: ${NUM_WORKERS})"
[[ "${SEED}" =~ ^[0-9]+$ ]] \
  || die "SEED must be a non-negative integer (received: ${SEED})"
[[ "${WAVELENGTH}" =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || die "WAVELENGTH must be a positive decimal value"
[[ "${MAX_FORMULA_DEVIATION}" =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || die "MAX_FORMULA_DEVIATION must be a non-negative decimal value"
[[ ${#STAGE_LOOP_ARGS[@]} -eq 3 ]] \
  || die "STAGE_LOOPS must contain exactly three positive integers"
for loop_count in "${STAGE_LOOP_ARGS[@]}"; do
  [[ "${loop_count}" =~ ^[1-9][0-9]*$ ]] \
    || die "STAGE_LOOPS must contain exactly three positive integers"
done
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] \
  || die "DRY_RUN must be 0 or 1"
[[ -x "${PYTHON_BIN}" ]] || die "XRD Python is not executable: ${PYTHON_BIN}"
[[ -f "${PROJECT_DIR}/QL_yfs_XRD.py" ]] \
  || die "QL entry point not found: ${PROJECT_DIR}/QL_yfs_XRD.py"
[[ -f "${PROJECT_DIR}/validate_occ_sweep_inputs.py" ]] \
  || die "input validator not found: ${PROJECT_DIR}/validate_occ_sweep_inputs.py"

selected_indices=()
for index in "${!SAMPLE_NAMES[@]}"; do
  if sample_is_selected "${SAMPLE_NAMES[index]}"; then
    selected_indices+=("${index}")
  fi
done
[[ ${#selected_indices[@]} -gt 0 ]] || die "sample filter selected no tasks"

# Validate every selected source before creating output directories or starting
# any CUDA task. This intentionally rejects the currently inconsistent ICSD
# 192731 CIF until a composition-compatible replacement is supplied.
printf 'Preflight validation (%d selected sample(s))\n' "${#selected_indices[@]}"
for index in "${selected_indices[@]}"; do
  source_dir="${OCC_DIR}/${SOURCE_DIRS[index]}"
  xy_file="${source_dir}/${XY_FILES[index]}"
  main_cif="${source_dir}/data.cif"
  "${PYTHON_BIN}" "${PROJECT_DIR}/validate_occ_sweep_inputs.py" \
    --sample "${SAMPLE_NAMES[index]}" \
    --physical-source-dir "${source_dir}" \
    --cif "${main_cif}" \
    --xy "${xy_file}" \
    --target "${STOICH_TARGETS[index]}" \
    --anion "${ANIONS[index]}" \
    --expected-marker "${IDENTITY_MARKERS[index]}" \
    --max-formula-deviation "${MAX_FORMULA_DEVIATION}" \
    || die "input preflight failed for ${SAMPLE_NAMES[index]}"
done

# Physical GPU 0 is exposed to PyTorch as logical cuda:0.
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" -c \
    'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; assert torch.cuda.device_count() == 1, "expected exactly one visible GPU"' \
    || die "Python/PyTorch cannot use CUDA GPU 0"
  [[ ! -e "${OUTPUT_ROOT}" ]] \
    || die "refusing to overwrite existing output root: ${OUTPUT_ROOT}"
  mkdir -p -- "${OUTPUT_ROOT}"
  {
    printf 'git_head=%s\n' "$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
    printf 'qlearning_sha256='
    sha256sum "${PROJECT_DIR}/src/yfs_xrd_refinement/qlearning.py" | awk '{print $1}'
    printf 'runner_sha256='
    sha256sum "${PROJECT_DIR}/run_occ_lambda_sweep_cuda0.sh" | awk '{print $1}'
    printf 'validator_sha256='
    sha256sum "${PROJECT_DIR}/validate_occ_sweep_inputs.py" | awk '{print $1}'
    printf 'wavelength_angstrom=%s\n' "${WAVELENGTH}"
    printf 'stage_loops=%s\n' "${STAGE_LOOPS}"
    printf 'seed=%s\n' "${SEED}"
    printf 'sample_filter=%s\n' "${SAMPLE_FILTER:-all}"
  } > "${OUTPUT_ROOT}/run_manifest.txt"
fi

printf 'Occupancy lambda sweep\n'
printf '  GPU:             physical CUDA 0\n'
printf '  scheduling:      sequential (one refinement at a time)\n'
printf '  lambdas:         %s\n' "${LAMBDAS[*]}"
printf '  stage loops:     %s\n' "${STAGE_LOOP_ARGS[*]}"
printf '  seed:            %s\n' "${SEED}"
printf '  wavelength:      %s Angstrom (Cu target)\n' "${WAVELENGTH}"
printf '  CPU workers:     %s\n' "${NUM_WORKERS}"
printf '  sample filter:   %s\n' "${SAMPLE_FILTER:-all four samples}"
printf '  output root:     %s\n' "${OUTPUT_ROOT}"

completed=0
failed=0
selected=0

for lambda_stoich in "${LAMBDAS[@]}"; do
  lambda_label="${lambda_stoich/./p}"

  for index in "${selected_indices[@]}"; do
    sample_name="${SAMPLE_NAMES[index]}"
    selected=$((selected + 1))
    source_dir="${OCC_DIR}/${SOURCE_DIRS[index]}"
    source_xy="${source_dir}/${XY_FILES[index]}"
    source_cif="${source_dir}/data.cif"
    stoich_target="${STOICH_TARGETS[index]}"
    run_dir="${OUTPUT_ROOT}/lambda_${lambda_label}/${sample_name}"
    input_dir="${run_dir}/inputs"
    snapshot_xy="${input_dir}/obs.xy"
    snapshot_cif="${input_dir}/data.cif"
    summary_file="${run_dir}/run_summary.json"

    command=(
      "${PYTHON_BIN}" "${PROJECT_DIR}/QL_yfs_XRD.py"
      --xy "${snapshot_xy}"
      --main "${snapshot_cif}"
      --wl "${WAVELENGTH}"
      --single-phase
      --stoich-phase data.cif
      --stoich "${stoich_target}"
      --lambda-stoich "${lambda_stoich}"
      --occupancy-objective stoich
      --stage-loops "${STAGE_LOOP_ARGS[@]}"
      --num-workers "${NUM_WORKERS}"
      --seed "${SEED}"
    )

    printf '\nRUN: lambda=%s sample=%s\n' "${lambda_stoich}" "${sample_name}"
    printf '  physical source: %s\n' "${source_dir}"
    printf '  target:          %s\n' "${stoich_target}"
    printf '  output:          %s\n' "${run_dir}"
    printf '  command:'
    printf ' %q' "${command[@]}"
    printf '\n'

    if [[ "${DRY_RUN}" == "1" ]]; then
      continue
    fi

    [[ ! -e "${run_dir}" ]] || die "refusing to overwrite run directory: ${run_dir}"
    mkdir -p -- "${input_dir}"
    cp -p -- "${source_xy}" "${snapshot_xy}"
    cp -p -- "${source_cif}" "${snapshot_cif}"
    "${PYTHON_BIN}" "${PROJECT_DIR}/validate_occ_sweep_inputs.py" \
      --sample "${sample_name}" \
      --physical-source-dir "${source_dir}" \
      --cif "${snapshot_cif}" \
      --xy "${snapshot_xy}" \
      --target "${stoich_target}" \
      --anion "${ANIONS[index]}" \
      --expected-marker "${IDENTITY_MARKERS[index]}" \
      --max-formula-deviation "${MAX_FORMULA_DEVIATION}" \
      --metadata-output "${run_dir}/input_manifest.json"
    {
      printf '#!/usr/bin/env bash\n'
      printf ' %q' "${command[@]}"
      printf '\n'
    } > "${run_dir}/run_command.sh"
    chmod +x "${run_dir}/run_command.sh"

    if (
      cd -- "${run_dir}"
      "${command[@]}" > train.log 2>&1
    ); then
      if [[ -s "${summary_file}" ]]; then
        printf 'SUCCESS\n' > "${run_dir}/task_status.txt"
        completed=$((completed + 1))
      else
        printf 'FAILED_INCOMPLETE_OUTPUT\n' > "${run_dir}/task_status.txt"
        printf 'FAILED: lambda=%s sample=%s (missing run_summary.json)\n' \
          "${lambda_stoich}" "${sample_name}" >&2
        failed=$((failed + 1))
      fi
    else
      printf 'FAILED_EXIT_NONZERO\n' > "${run_dir}/task_status.txt"
      printf 'FAILED: lambda=%s sample=%s (see %s/train.log)\n' \
        "${lambda_stoich}" "${sample_name}" "${run_dir}" >&2
      failed=$((failed + 1))
    fi
  done
done

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '\nDry run complete: %d commands validated; no refinement was started.\n' "${selected}"
else
  {
    printf 'selected=%d\n' "${selected}"
    printf 'succeeded=%d\n' "${completed}"
    printf 'failed=%d\n' "${failed}"
  } > "${OUTPUT_ROOT}/run_status.txt"
  printf '\nSweep complete: %d succeeded, %d failed.\n' "${completed}" "${failed}"
  (( failed == 0 )) || exit 1
fi
