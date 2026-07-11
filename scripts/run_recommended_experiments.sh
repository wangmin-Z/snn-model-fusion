#!/usr/bin/env bash

# 按推荐顺序完成 ResNet-18、80/20 划分、seed=42 的论文复现实验。
# 脚本会优先读取 latest checkpoint，因此中途暂停后可直接再次运行。

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DRY_RUN=false
case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=true ;;
  --help|-h)
    echo "用法: $0 [--dry-run]"
    echo "  --dry-run  只打印实验命令，不检查数据或启动训练。"
    exit 0
    ;;
  *)
    echo "未知参数: $1" >&2
    echo "请使用 --help 查看用法。" >&2
    exit 2
    ;;
esac

# 优先使用当前已激活环境，其次使用项目同级的 deep-learning-study/.venv。
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${PROJECT_ROOT}/../.venv/bin/python" ]]; then
  PYTHON="${PROJECT_ROOT}/../.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

OUTPUT_ROOT="outputs/paper_recommended_20260711"

UCM_ROOT="data/processed/UCMerced_LandUse/UCMerced_LandUse/Images"
RSSCN7_ROOT="data/processed/RSSCN7/RSSCN7-master"
AID_ROOT="data/processed/AID"

UCM_ANN="outputs/paper_full_120_20260704/ucm80_ann_resnet18_img224_e120_seed42/best_ann_resnet18.pt"
RSSCN7_ANN="outputs/paper_full_120_20260704/rsscn7_80_ann_resnet18_img224_e120_seed42/best_ann_resnet18.pt"
AID_ANN="${OUTPUT_ROOT}/aid80_ann_resnet18_img224_e120_seed42/best_ann_resnet18.pt"

COMMON_ARGS=(
  --depth 18
  --train-ratio 0.8
  --epochs 120
  --batch-size 8
  --image-size 224
  --num-workers 0
  --device mps
  --lr 0.1
  --momentum 0.9
  --weight-decay 0
  --scheduler-step 10
  --scheduler-gamma 0.3
  --seed 42
  --disable-progress
)

check_required_path() {
  if [[ ! -e "$1" ]]; then
    echo "缺少必要路径: $1" >&2
    exit 1
  fi
}

run_train() {
  if [[ "${DRY_RUN}" == true ]]; then
    printf '将运行:'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

run_ann() {
  local data_root="$1"
  local run_name="$2"
  local latest="${OUTPUT_ROOT}/${run_name}/latest_ann_resnet18.pt"
  local resume_args=()

  if [[ "${DRY_RUN}" == false && -f "${latest}" ]]; then
    resume_args=(--resume-checkpoint "${latest}")
  else
    resume_args=(--pretrained-imagenet)
  fi

  echo "开始 ANN 实验: ${run_name}"
  run_train "${PYTHON}" train.py \
    --model-type ann \
    --data-root "${data_root}" \
    --output-dir "${OUTPUT_ROOT}" \
    --run-name "${run_name}" \
    "${COMMON_ARGS[@]}" \
    "${resume_args[@]}"
}

run_snn() {
  local data_root="$1"
  local run_name="$2"
  local time_steps="$3"
  local ann_checkpoint="$4"
  local latest="${OUTPUT_ROOT}/${run_name}/latest_snn_resnet18.pt"
  local init_args=()

  if [[ "${DRY_RUN}" == false && -f "${latest}" ]]; then
    init_args=(--resume-checkpoint "${latest}")
  else
    if [[ "${DRY_RUN}" == false ]]; then
      check_required_path "${ann_checkpoint}"
    fi
    init_args=(--ann-checkpoint "${ann_checkpoint}")
  fi

  echo "开始 SNN 实验: ${run_name}"
  run_train "${PYTHON}" train.py \
    --model-type snn \
    --time-steps "${time_steps}" \
    --data-root "${data_root}" \
    --output-dir "${OUTPUT_ROOT}" \
    --run-name "${run_name}" \
    "${COMMON_ARGS[@]}" \
    "${init_args[@]}"
}

if [[ "${DRY_RUN}" == false ]]; then
  check_required_path "${UCM_ROOT}"
  check_required_path "${RSSCN7_ROOT}"
  check_required_path "${AID_ROOT}"
  check_required_path "${UCM_ANN}"
  check_required_path "${RSSCN7_ANN}"
fi

# 1. 先补完只剩 17 轮的 UCM T=8。
run_train "${PYTHON}" train.py \
  --model-type snn \
  --time-steps 8 \
  --data-root "${UCM_ROOT}" \
  --resume-checkpoint "outputs/paper_full_120_20260706/ucm80_snn_mft_resnet18_T8_img224_e120_seed42/latest_snn_resnet18.pt" \
  --output-dir "outputs/paper_full_120_20260706" \
  --run-name "ucm80_snn_mft_resnet18_T8_img224_e120_seed42" \
  "${COMMON_ARGS[@]}"

# 2. 完成 RSSCN7 尚缺的三个时间窗口。
run_snn "${RSSCN7_ROOT}" "rsscn7_80_snn_mft_resnet18_T2_img224_e120_seed42" 2 "${RSSCN7_ANN}"
run_snn "${RSSCN7_ROOT}" "rsscn7_80_snn_mft_resnet18_T6_img224_e120_seed42" 6 "${RSSCN7_ANN}"
run_snn "${RSSCN7_ROOT}" "rsscn7_80_snn_mft_resnet18_T8_img224_e120_seed42" 8 "${RSSCN7_ANN}"

# 3. 训练 AID 的 ANN，再复制其卷积权重训练四个 SNN 时间窗口。
run_ann "${AID_ROOT}" "aid80_ann_resnet18_img224_e120_seed42"
run_snn "${AID_ROOT}" "aid80_snn_mft_resnet18_T2_img224_e120_seed42" 2 "${AID_ANN}"
run_snn "${AID_ROOT}" "aid80_snn_mft_resnet18_T4_img224_e120_seed42" 4 "${AID_ANN}"
run_snn "${AID_ROOT}" "aid80_snn_mft_resnet18_T6_img224_e120_seed42" 6 "${AID_ANN}"
run_snn "${AID_ROOT}" "aid80_snn_mft_resnet18_T8_img224_e120_seed42" 8 "${AID_ANN}"

if [[ "${DRY_RUN}" == true ]]; then
  echo "实验队列预览结束，未启动训练。"
else
  echo "推荐实验队列已全部完成。"
fi
