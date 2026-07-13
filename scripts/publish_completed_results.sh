#!/usr/bin/env bash

# 监控推荐实验队列；每个实验完成 120 轮后，归档轻量结果并推送当前发布分支。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SOURCE_ROOT=""
INTERVAL_SECONDS=300
ONCE=false

usage() {
  cat <<'EOF'
用法: scripts/publish_completed_results.sh --source-root /path/to/training/repo [选项]

选项:
  --interval-seconds N  轮询间隔，默认 300 秒。
  --once                只检查一次，不持续轮询。
  --help                显示本帮助。
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-root) SOURCE_ROOT="$2"; shift 2 ;;
    --interval-seconds) INTERVAL_SECONDS="$2"; shift 2 ;;
    --once) ONCE=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${SOURCE_ROOT}" || ! -d "${SOURCE_ROOT}" ]]; then
  echo "--source-root 必须指向正在训练的仓库目录。" >&2
  exit 2
fi

if ! [[ "${INTERVAL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--interval-seconds 必须为正整数。" >&2
  exit 2
fi

PYTHON="${PYTHON:-python3}"

# 数据集|模型|时间步|相对于训练仓库的输出目录
RUNS=(
  "RSSCN7|SNN-MFT ResNet-18|2|outputs/paper_recommended_20260711/rsscn7_80_snn_mft_resnet18_T2_img224_e120_seed42"
  "RSSCN7|SNN-MFT ResNet-18|6|outputs/paper_recommended_20260711/rsscn7_80_snn_mft_resnet18_T6_img224_e120_seed42"
  "RSSCN7|SNN-MFT ResNet-18|8|outputs/paper_recommended_20260711/rsscn7_80_snn_mft_resnet18_T8_img224_e120_seed42"
  "AID|ANN ResNet-18|-|outputs/paper_recommended_20260711/aid80_ann_resnet18_img224_e120_seed42"
  "AID|SNN-MFT ResNet-18|2|outputs/paper_recommended_20260711/aid80_snn_mft_resnet18_T2_img224_e120_seed42"
  "AID|SNN-MFT ResNet-18|4|outputs/paper_recommended_20260711/aid80_snn_mft_resnet18_T4_img224_e120_seed42"
  "AID|SNN-MFT ResNet-18|6|outputs/paper_recommended_20260711/aid80_snn_mft_resnet18_T6_img224_e120_seed42"
  "AID|SNN-MFT ResNet-18|8|outputs/paper_recommended_20260711/aid80_snn_mft_resnet18_T8_img224_e120_seed42"
)

publish_if_complete() {
  local dataset="$1"
  local model="$2"
  local time_steps="$3"
  local output_path="$4"
  local source_run="${SOURCE_ROOT}/${output_path}"
  local run_name
  run_name="$(basename "${source_run}")"

  [[ -f "${source_run}/history.csv" ]] || return 0
  [[ -d "results/completed/${run_name}" ]] && return 0

  local last_epoch
  last_epoch="$(tail -n 1 "${source_run}/history.csv" | cut -d, -f1)"
  [[ "${last_epoch}" == "120" ]] || return 0

  local archive_args=(
    --source-run "${source_run}"
    --dataset "${dataset}"
    --model "${model}"
    --output-path "${output_path}"
  )
  if [[ "${time_steps}" != "-" ]]; then
    archive_args+=(--time-steps "${time_steps}")
  fi

  "${PYTHON}" scripts/archive_completed_experiment.py "${archive_args[@]}"
  git add "results/completed/${run_name}" results/completed/README.md docs/results.md
  git diff --cached --check
  git commit -m "archive ${run_name} result"
  git -c credential.helper= -c 'credential.helper=!gh auth git-credential' -c http.version=HTTP/1.1 push
  echo "已推送完成实验: ${run_name}"
}

while true; do
  for entry in "${RUNS[@]}"; do
    IFS='|' read -r dataset model time_steps output_path <<< "${entry}"
    publish_if_complete "${dataset}" "${model}" "${time_steps}" "${output_path}"
  done

  [[ "${ONCE}" == true ]] && break
  sleep "${INTERVAL_SECONDS}"
done
