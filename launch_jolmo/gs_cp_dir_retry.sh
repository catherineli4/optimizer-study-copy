#!/usr/bin/env bash
# Compact GCS directory download with flock + retries.
# Called as one short line from SLURM array scripts so retries are not unrolled
# into every task arm (which hits Babel's batch-script size / Pathname limit).
#
# Usage:
#   gs_cp_dir_retry.sh <gs_dir> <local_dir> <max_attempts> <base_delay> [required_file ...]
set -euo pipefail

gs_dir="${1:?gs_dir required}"
local_dir="${2:?local_dir required}"
max_attempts="${3:?max_attempts required}"
base_delay="${4:?base_delay required}"
shift 4
required_files=("$@")

dest="${local_dir%/}"
src="${gs_dir%/}/*"
path_hash="$(printf '%s' "$dest" | sha256sum | cut -c1-10)"
lockfile="/tmp/${path_hash}.lock"

complete() {
  if ((${#required_files[@]})); then
    local f
    for f in "${required_files[@]}"; do
      [[ -f "$f" ]] || return 1
    done
    return 0
  fi
  [[ -d "$dest" ]] && ls -A "$dest" 2>/dev/null | grep -q .
}

attempt_download() {
  local i="$1"
  local state_dir="/tmp/gsutil-state-${path_hash}-a${i}"
  rm -rf "$dest" "$state_dir"
  mkdir -p "$dest" "$state_dir"
  gsutil -o "GSUtil:sliced_object_download_threshold=0" \
    -o "GSUtil:parallel_thread_count=1" \
    -o "GSUtil:parallel_process_count=1" \
    -o "GSUtil:state_dir=${state_dir}" \
    cp -r "$src" "$dest"/
  complete
  rm -rf "$state_dir"
}

(
  flock -x 9
  if complete; then
    echo "Skipping download, ${dest} already complete"
    exit 0
  fi
  delay="$base_delay"
  for ((i = 0; i < max_attempts; i++)); do
    if ((i > 0)); then
      sleep "$delay"
      delay=$((delay * 2))
    fi
    if attempt_download "$i"; then
      exit 0
    fi
  done
  echo "download failed after ${max_attempts} attempts: ${dest}"
  rm -rf "$dest"
  exit 1
) 9>"$lockfile"
