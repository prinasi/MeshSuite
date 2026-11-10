#!/usr/bin/env bash
set -uo pipefail

# Re-run every engine deployment evaluation (Std. and Cust.) for every method,
# method variant and dataset, then merge the results into one text summary.
#
# Each scene runs through single_unity_eval.sh with all stages forced (export or
# DiffSoup bake, Unity capture, Player profile, image metrics), so no number is
# carried over from an earlier campaign. One report is written per
# method x setting x dataset from the scenes that completed in this campaign,
# and tools/summarize_deployment_metrics.py merges them.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

ALL_METHODS=(triangle-splatting triangle-splatting-game-engine 2dts 2dts-opaque diffsoup mesh-splatting)
ALL_DATASETS=(mipnerf360 tandt nerf_synthetic)
ALL_CONDITIONS=(standard-engine custom-engine)
MIPNERF360_SCENES=(bicycle flowers garden stump treehill room counter kitchen bonsai)
TANDT_SCENES=(truck train)
NERF_SYNTHETIC_SCENES=(chair drums ficus hotdog lego materials mic ship)

METHODS=()
DATASETS_SEL=()
CONDITIONS=()
GPU_ID=""
EVAL_ROOT="outputs/deployment_eval"
CONFIG_ROOT="configs"
RESUME=0
SUMMARY_ONLY=0
DRY_RUN=0
RETRIES=2
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-}}"
UNITY_BIN="${UNITY:-}"
UNITY_PROJECT_DIR="${PROJECT:-${UNITY_PROJECT:-}}"
UNITY_DISPLAY="${UNITY_DISPLAY:-${DISPLAY:-}}"
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage:
  bash $0 <gpu_id> --unity PATH --unity-project PATH [options] [-- single_unity_eval.sh options]

Runs single_unity_eval.sh for every scene of every selected method x dataset x
setting with all stages forced, writes one report per group, and merges them into
  ${EVAL_ROOT}/deployment_metrics_summary.txt  (plus a full-precision .csv).

Defaults cover:
  methods   ${ALL_METHODS[*]}
  datasets  ${ALL_DATASETS[*]}
  settings  ${ALL_CONDITIONS[*]}
Groups without configs (2dts-opaque on nerf_synthetic) are skipped.

Options:
  --unity PATH           Unity Editor executable (default: \$UNITY)
  --unity-project PATH   Unity project (default: \$PROJECT or \$UNITY_PROJECT)
  --display DISPLAY      X display for the Editor and Player (default: \$DISPLAY)
  --methods "A B"        Subset of methods (space- or comma-separated)
  --datasets "A B"       Subset of datasets
  --settings "A B"       Subset of settings: standard-engine, custom-engine (std/cust accepted)
  --eval-root PATH       Campaign directory: status, logs, reports, summary (default: ${EVAL_ROOT})
  --config-root PATH     Config root (default: ${CONFIG_ROOT})
  --resume               Keep this campaign's completed scenes and run only the rest
                         (they still run with all stages forced)
  --retries N            Re-run a failed scene up to N more times (default: ${RETRIES});
                         the standalone Player occasionally segfaults on shutdown
  --summary-only         Only rebuild the summary from existing group reports
  --dry-run              Print the commands without running them
  --python PATH          Python executable passed to single_unity_eval.sh
  -h, --help             Show this help

Everything after "--" is passed to every single_unity_eval.sh call (for example
--profile-runs 3 --player-batchmode); pass it unchanged to a --resume run so the
report step sees the same profile settings.

Without --resume, the selected groups' completion markers and reports are cleared
first and the standalone Player is rebuilt once (--force-player-build), so FPS is
never measured on stale shaders. Scene outputs are overwritten in place.
EOF
}

require_value() { [[ "$#" -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }; }
split_list() { printf '%s' "$1" | tr ',' ' '; }

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --unity) require_value "$@"; UNITY_BIN="$2"; shift 2 ;;
        --unity-project|--unity_project|--project) require_value "$@"; UNITY_PROJECT_DIR="$2"; shift 2 ;;
        --display) require_value "$@"; UNITY_DISPLAY="$2"; shift 2 ;;
        --methods) require_value "$@"; read -r -a METHODS <<<"$(split_list "$2")"; shift 2 ;;
        --datasets) require_value "$@"; read -r -a DATASETS_SEL <<<"$(split_list "$2")"; shift 2 ;;
        --settings|--conditions) require_value "$@"; read -r -a CONDITIONS <<<"$(split_list "$2")"; shift 2 ;;
        --eval-root|--eval_root) require_value "$@"; EVAL_ROOT="$2"; shift 2 ;;
        --config-root|--config_root) require_value "$@"; CONFIG_ROOT="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --retries) require_value "$@"; RETRIES="$2"; shift 2 ;;
        --summary-only|--summary_only) SUMMARY_ONLY=1; shift ;;
        --dry-run|--dry_run) DRY_RUN=1; shift ;;
        --python) require_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        --*) echo "Unknown option: $1" >&2; usage; exit 1 ;;
        *)
            [[ -z "${GPU_ID}" ]] || { echo "Unexpected argument: $1" >&2; exit 1; }
            GPU_ID="$1"; shift ;;
    esac
done

[[ "${#METHODS[@]}" -gt 0 ]] || METHODS=("${ALL_METHODS[@]}")
[[ "${#DATASETS_SEL[@]}" -gt 0 ]] || DATASETS_SEL=("${ALL_DATASETS[@]}")
[[ "${#CONDITIONS[@]}" -gt 0 ]] || CONDITIONS=("${ALL_CONDITIONS[@]}")
for i in "${!CONDITIONS[@]}"; do
    case "${CONDITIONS[$i]}" in
        std|standard|standard-engine|standard_engine) CONDITIONS[$i]="standard-engine" ;;
        cust|custom|custom-engine|custom_engine) CONDITIONS[$i]="custom-engine" ;;
        *) echo "Unknown setting: ${CONDITIONS[$i]}" >&2; exit 1 ;;
    esac
done
[[ "${RETRIES}" =~ ^[0-9]+$ ]] || { echo "--retries must be a non-negative integer" >&2; exit 1; }

if [[ -z "${PYTHON_BIN}" ]]; then
    if [[ -x "/opt/miniconda3/envs/torch/bin/python" ]]; then PYTHON_BIN="/opt/miniconda3/envs/torch/bin/python"
    elif command -v python >/dev/null 2>&1; then PYTHON_BIN="python"
    else PYTHON_BIN="python3"; fi
fi

STATUS_DIR="${EVAL_ROOT}/status"
LOG_DIR="${EVAL_ROOT}/logs"
REPORTS_DIR="${EVAL_ROOT}/reports"
SUMMARY_TXT="${EVAL_ROOT}/deployment_metrics_summary.txt"

summarize() {
    local cmd=("${PYTHON_BIN}" tools/summarize_deployment_metrics.py
        --reports-root "${REPORTS_DIR}" --config-root "${CONFIG_ROOT}"
        --methods "${METHODS[@]}" --datasets "${DATASETS_SEL[@]}" --conditions "${CONDITIONS[@]}"
        --output "${SUMMARY_TXT}")
    printf '[summary] Command:'; printf ' %q' "${cmd[@]}"; printf '\n'
    [[ "${DRY_RUN}" -eq 1 ]] || "${cmd[@]}"
}

if [[ "${SUMMARY_ONLY}" -eq 1 ]]; then
    summarize
    exit $?
fi

[[ -n "${GPU_ID}" ]] || { usage; exit 1; }
[[ -n "${UNITY_BIN}" ]] || { echo "Unity executable is required; pass --unity or set UNITY." >&2; exit 1; }
[[ -n "${UNITY_PROJECT_DIR}" ]] || { echo "Unity project is required; pass --unity-project or set PROJECT." >&2; exit 1; }
if [[ -z "${UNITY_DISPLAY}" ]]; then
    echo "Warning: no display set; the Player profile needs one for GPU timings (pass --display)." >&2
fi

scenes_for() {
    case "$1" in
        mipnerf360) echo "${MIPNERF360_SCENES[*]}" ;;
        tandt) echo "${TANDT_SCENES[*]}" ;;
        nerf_synthetic) echo "${NERF_SYNTHETIC_SCENES[*]}" ;;
        *) echo "Unknown dataset: $1" >&2; return 1 ;;
    esac
}

# The Unity package a run deploys; Std. and Cust. share unity_native, so it is
# exported once per campaign. DiffSoup's Std. package is its texture bake.
asset_subdir_for() {
    if [[ "$1" == "diffsoup" && "$2" == "standard-engine" ]]; then echo "unity_baked"; else echo "unity_native"; fi
}

run_logged() {
    local log="$1"; shift
    printf '[run] Command:'; printf ' %q' "$@"; printf '\n'
    # A dry run still lets single_unity_eval.sh resolve every config and path.
    [[ "${DRY_RUN}" -eq 0 ]] || { "$@" --dry-run; return; }
    mkdir -p "$(dirname "${log}")"
    "$@" 2>&1 | tee -a "${log}"
    return "${PIPESTATUS[0]}"
}

COMMON_ARGS=(--unity "${UNITY_BIN}" --unity-project "${UNITY_PROJECT_DIR}"
    --config-root "${CONFIG_ROOT}" --python "${PYTHON_BIN}")
# --display reaches the Player; the Editor capture reads DISPLAY from the environment.
if [[ -n "${UNITY_DISPLAY}" ]]; then
    COMMON_ARGS+=(--display "${UNITY_DISPLAY}")
    export DISPLAY="${UNITY_DISPLAY}"
fi

# Clear the selected groups' state so stale results never reach the summary.
if [[ "${RESUME}" -eq 0 && "${DRY_RUN}" -eq 0 ]]; then
    rm -f "${STATUS_DIR}/player_built"
    for method in "${METHODS[@]}"; do
        for dataset in "${DATASETS_SEL[@]}"; do
            rm -rf "${STATUS_DIR}/exported/${method}/${dataset}"
            for condition in "${CONDITIONS[@]}"; do
                rm -rf "${STATUS_DIR}/done/${method}/${condition}/${dataset}" \
                       "${REPORTS_DIR}/${method}/${condition}/${dataset}"
            done
        done
    done
fi

GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
[[ -z "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]] || GIT_COMMIT="${GIT_COMMIT}-dirty"
PROFILE_RUNTIME="player"
for i in "${!EXTRA_ARGS[@]}"; do
    [[ "${EXTRA_ARGS[$i]}" == --profile-runtime || "${EXTRA_ARGS[$i]}" == --profile_runtime ]] \
        && PROFILE_RUNTIME="${EXTRA_ARGS[$((i + 1))]:-player}"
done
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU_ID}" 2>/dev/null | head -n1 || true)"
FAILURES=()

for method in "${METHODS[@]}"; do
    for dataset in "${DATASETS_SEL[@]}"; do
        if [[ ! -d "${CONFIG_ROOT}/${method}/${dataset}" ]]; then
            echo "[skip] ${method}/${dataset}: no configs"
            continue
        fi
        read -r -a scenes <<<"$(scenes_for "${dataset}")"
        for condition in "${CONDITIONS[@]}"; do
            group="${method}/${condition}/${dataset}"
            asset_subdir="$(asset_subdir_for "${method}" "${condition}")"
            done_targets=()
            for scene in "${scenes[@]}"; do
                [[ -f "${CONFIG_ROOT}/${method}/${dataset}/${scene}.yaml" ]] || continue
                done_marker="${STATUS_DIR}/done/${group}/${scene}"
                if [[ -f "${done_marker}" ]]; then
                    echo "[${group}/${scene}] Completed in this campaign; skipping"
                    done_targets+=("${dataset}/${scene}")
                    continue
                fi
                export_marker="${STATUS_DIR}/exported/${method}/${dataset}/${scene}.${asset_subdir}"
                log="${LOG_DIR}/${group}/${scene}.log"
                [[ "${DRY_RUN}" -eq 1 ]] || rm -f "${log}"
                ok=0
                for attempt in $(seq 0 "${RETRIES}"); do
                    flags=(--force-capture --force-profile --force-metrics)
                    [[ -f "${export_marker}" ]] || flags+=(--force-export)
                    [[ -f "${STATUS_DIR}/player_built" ]] || flags+=(--force-player-build)
                    echo "[${group}/${scene}] attempt $((attempt + 1))/$((RETRIES + 1))"
                    if run_logged "${log}" bash single_unity_eval.sh "${method}" "${dataset}/${scene}" "${GPU_ID}" \
                        "--${condition}" "${COMMON_ARGS[@]}" "${flags[@]}" \
                        --report-dir "${EVAL_ROOT}/scratch_reports/${group}/${scene}" \
                        "${EXTRA_ARGS[@]}"; then
                        ok=1
                        break
                    fi
                    # A failure after the Player build or export must not repeat them.
                    if [[ "${DRY_RUN}" -eq 0 ]] && grep -q "\[unity-player-build\]\|Existing standalone Player found" "${log}" \
                        && ! grep -q "Standalone Player build failed" "${log}"; then
                        mkdir -p "${STATUS_DIR}"; touch "${STATUS_DIR}/player_built"
                    fi
                done
                if [[ "${ok}" -eq 1 ]]; then
                    if [[ "${DRY_RUN}" -eq 0 ]]; then
                        mkdir -p "$(dirname "${done_marker}")" "$(dirname "${export_marker}")"
                        date -Is > "${done_marker}"
                        touch "${export_marker}" "${STATUS_DIR}/player_built"
                    fi
                    done_targets+=("${dataset}/${scene}")
                else
                    echo "[${group}/${scene}] FAILED after $((RETRIES + 1)) attempts; see ${log}" >&2
                    FAILURES+=("${group}/${scene}")
                fi
            done

            if [[ "${#done_targets[@]}" -eq 0 ]]; then
                echo "[${group}] No completed scenes; no report"
                continue
            fi
            report_dir="${REPORTS_DIR}/${group}"
            # Only this campaign's completed scenes enter the report, never older outputs.
            if run_logged "${LOG_DIR}/${group}/report.log" bash single_unity_eval.sh "${method}" "${done_targets[@]}" "${GPU_ID}" \
                "--${condition}" --report-only --config-root "${CONFIG_ROOT}" --python "${PYTHON_BIN}" \
                --report-dir "${report_dir}" "${EXTRA_ARGS[@]}"; then
                if [[ "${DRY_RUN}" -eq 0 ]]; then
                    "${PYTHON_BIN}" - "${report_dir}/run_info.json" "${GIT_COMMIT}" "${GPU_NAME}" "${UNITY_BIN}" "${PROFILE_RUNTIME}" "${done_targets[@]}" <<'PY'
import json
import sys
from datetime import datetime
path, commit, gpu, unity, runtime, *targets = sys.argv[1:]
json.dump({
    "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
    "git_commit": commit,
    "gpu": gpu or None,
    "unity": unity,
    "profile_runtime": runtime,
    "scenes": targets,
}, open(path, "w"), indent=2)
PY
                fi
            else
                FAILURES+=("${group}:report")
            fi
        done
    done
done

summarize

if [[ "${#FAILURES[@]}" -gt 0 ]]; then
    echo "" >&2
    echo "Failed (${#FAILURES[@]}); fix and re-run with --resume (same options) to finish:" >&2
    printf '  %s\n' "${FAILURES[@]}" >&2
    exit 1
fi
echo "All deployment evaluations complete: ${SUMMARY_TXT}"
