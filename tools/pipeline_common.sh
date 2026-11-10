# Shared aliases for the existing repository shell entry points.
# Unknown names pass through so future method configs remain usable.
canonical_method() {
    local name
    name="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
    name="${name//_/-}"
    case "${name}" in
        trianglesplatting|ts) echo "triangle-splatting" ;;
        trianglesplatting-game-engine|ts-game-engine) echo "triangle-splatting-game-engine" ;;
        meshsplatting|ms) echo "mesh-splatting" ;;
        t2-splat) echo "t2splat" ;;
        d2ts) echo "2dts" ;;
        d2ts-opaque) echo "2dts-opaque" ;;
        diff-soup) echo "diffsoup" ;;
        *) echo "${name}" ;;
    esac
}

# Method variants selected by a command-line flag. A variant has no scene configs
# of its own: it runs its base method's scene config followed by
# configs/base/<variant>-<dataset>.yaml, while the variant name labels its output
# tree (output.method), reports and logs. Mirrors METHOD_VARIANTS in
# msbench/methods.py.
variant_base_method() {
    case "$1" in
        2dts-opaque) echo "2dts" ;;
        *) return 1 ;;
    esac
}

# `--opaque` selects 2dts-opaque: 2DTS trained and deployed under the opaque
# strategy (configs/base/2dts-opaque.yaml).
opaque_method() {
    case "$1" in
        2dts|2dts-opaque) echo "2dts-opaque" ;;
        *) echo "--opaque applies only to 2dts, not $1" >&2; return 1 ;;
    esac
}

# Print the files METHOD's DATASET/SCENE config is made of, one per line: the
# scene config itself, or for a variant its base method's scene config and then
# the variant's dataset overlay. Mirrors scene_config_inputs() in msbench/methods.py.
scene_config_inputs() {
    local config_root="$1" method="$2" dataset="$3" scene="$4" base
    if base="$(variant_base_method "${method}")"; then
        echo "${config_root}/${base}/${dataset}/${scene}.yaml"
        echo "${config_root}/base/${method}-${dataset}.yaml"
    else
        echo "${config_root}/${method}/${dataset}/${scene}.yaml"
    fi
}

has_scene_config() {
    local input
    while IFS= read -r input; do
        [[ -f "${input}" ]] || return 1
    done < <(scene_config_inputs "$@")
}

# Print the scene config METHOD runs for DATASET/SCENE. A variant's config is
# generated into DIR with absolute bases; when one of its inputs is missing, that
# input's path is printed instead, so callers report it as the missing config.
scene_config_for() {
    local config_root="$1" method="$2" dataset="$3" scene="$4" dir="$5"
    local input output inputs=()
    while IFS= read -r input; do inputs+=("${input}"); done \
        < <(scene_config_inputs "${config_root}" "${method}" "${dataset}" "${scene}")
    if ! variant_base_method "${method}" >/dev/null; then
        echo "${inputs[0]}"
        return 0
    fi
    for input in "${inputs[@]}"; do
        [[ -f "${input}" ]] || { echo "${input}"; return 0; }
    done
    mkdir -p "${dir}"
    output="${dir}/${method}_${dataset}_${scene}.yaml"
    {
        echo "# Generated for ${method}: its base method's scene config, then the variant's dataset overlay."
        echo "_base_:"
        for input in "${inputs[@]}"; do
            echo "  - $(cd "$(dirname "${input}")" && pwd)/$(basename "${input}")"
        done
    } > "${output}"
    echo "${output}"
}

# A temporary directory for generated variant configs, removed when the script
# exits; empty when METHOD is not a variant. Scripts with their own EXIT trap pass
# their own directory to scene_config_for instead.
variant_config_dir() {
    variant_base_method "$1" >/dev/null || return 0
    mktemp -d "${TMPDIR:-/tmp}/msbench-variant-configs.XXXXXX"
}
