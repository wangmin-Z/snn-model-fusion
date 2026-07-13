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
REPOSITORY="wangmin-Z/snn-model-fusion"

github_get() {
  local endpoint="$1"
  local query="$2"
  local output
  for attempt in {1..8}; do
    if output="$(gh api "${endpoint}" --jq "${query}" 2>/tmp/snn_publish_api_error.log)"; then
      printf '%s' "${output}"
      return 0
    fi
    sleep 2
  done
  cat /tmp/snn_publish_api_error.log >&2
  return 1
}

github_json() {
  local method="$1"
  local endpoint="$2"
  local payload="$3"
  local query="$4"
  local output
  for attempt in {1..8}; do
    if output="$(printf '%s' "${payload}" | gh api --method "${method}" "${endpoint}" --input - --jq "${query}" 2>/tmp/snn_publish_api_error.log)"; then
      printf '%s' "${output}"
      return 0
    fi
    sleep 2
  done
  cat /tmp/snn_publish_api_error.log >&2
  return 1
}

publish_latest_commit() {
  local branch
  branch="$(git branch --show-current)"
  local remote_parent
  remote_parent="$(github_get "repos/${REPOSITORY}/git/ref/heads/${branch}" '.object.sha')"
  local base_tree
  base_tree="$(github_get "repos/${REPOSITORY}/git/commits/${remote_parent}" '.tree.sha')"
  local entries='[]'
  local path payload blob_sha

  while IFS= read -r path; do
    payload="$({ base64 < "${path}" | tr -d '\n'; } | jq -Rs '{content: ., encoding: "base64"}')"
    blob_sha="$(github_json POST "repos/${REPOSITORY}/git/blobs" "${payload}" '.sha')"
    entries="$(jq --arg path "${path}" --arg sha "${blob_sha}" '. + [{path: $path, mode: "100644", type: "blob", sha: $sha}]' <<< "${entries}")"
  done < <(git diff --name-only HEAD^ HEAD)

  payload="$(jq -n --arg base_tree "${base_tree}" --argjson tree "${entries}" '{base_tree: $base_tree, tree: $tree}')"
  local tree_sha
  tree_sha="$(github_json POST "repos/${REPOSITORY}/git/trees" "${payload}" '.sha')"
  local message
  message="$(git log -1 --format=%s)"
  payload="$(jq -n --arg message "${message}" --arg tree "${tree_sha}" --arg parent "${remote_parent}" '{message: $message, tree: $tree, parents: [$parent]}')"
  local commit_sha
  commit_sha="$(github_json POST "repos/${REPOSITORY}/git/commits" "${payload}" '.sha')"
  payload="$(jq -n --arg sha "${commit_sha}" '{sha: $sha, force: false}')"
  github_json PATCH "repos/${REPOSITORY}/git/refs/heads/${branch}" "${payload}" '.object.sha' >/dev/null
  echo "已通过 GitHub API 推送提交: ${commit_sha}"
}

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
  if git ls-files --error-unmatch "results/completed/${run_name}/history.csv" >/dev/null 2>&1; then
    return 0
  fi

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
  publish_latest_commit
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
