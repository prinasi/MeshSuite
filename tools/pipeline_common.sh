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
