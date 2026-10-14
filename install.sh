#!/usr/bin/env bash
# Install the vendored AirLLM patches into the active Python environment.
#
# The repository ships the *modified* AirLLM 4.0.0 source files under patches/.
# This script backs up the installed originals (with a timestamp) and copies the
# patched files in. Run it after `pip install airllm==4.0.0`.
#
# Usage:
#   ./install.sh
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_BASE="$(python3 -c 'import airllm, os; print(os.path.dirname(airllm.__file__))' | tail -n1)"

if [ -z "$PKG_BASE" ] || [ ! -d "$PKG_BASE" ]; then
    echo "ERROR: cannot locate the airllm package. Is airllm installed?" >&2
    exit 1
fi

TS="$(date +%Y%m%d%H%M%S)"
echo "AirLLM package directory: $PKG_BASE"
echo "Backup timestamp: $TS"

install_file() {
    local src="$1" dst="$2"
    cp "$dst" "$dst.bak.$TS"
    cp "$src" "$dst"
    echo "  installed $(basename "$src") -> $dst (backup: $dst.bak.$TS)"
}

install_file "$REPO_DIR/patches/airllm_base.py"    "$PKG_BASE/airllm_base.py"
install_file "$REPO_DIR/patches/utils.py"          "$PKG_BASE/utils.py"
install_file "$REPO_DIR/patches/persist/safetensor_model_persister.py" \
             "$PKG_BASE/persist/safetensor_model_persister.py"

# Clear compiled bytecode caches so the new sources are picked up.
find "$PKG_BASE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo
echo "Patched files installed."
echo "To revert: cp '$PKG_BASE/airllm_base.py.bak.$TS' '$PKG_BASE/airllm_base.py' (and same for the other two)."
echo "Remember to run every workload with AIRLLM_PREFETCH_WITH_COMPRESSION=1."