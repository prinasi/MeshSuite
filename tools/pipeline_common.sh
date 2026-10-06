# Shared aliases for the existing repository shell entry points.
# Unknown names pass through so future method configs remain usable.
canonical_method() {
    local name
    name="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
    name="${name//_/-}"
    case "${name}" in
        trianglesplatting|ts) echo "triangle-splatting" ;;
        meshsplatting|ms) echo "mesh-splatting" ;;
        d2ts) echo "2dts" ;;
        diff-soup) echo "diffsoup" ;;
        *) echo "${name}" ;;
    esac
}
