#!/usr/bin/env bash
# Open a gpsk_comms flowgraph in GNU Radio Companion, straight from the source
# tree -- nothing needs to be installed first.
#
#   ./run_grc.sh                        # anti-jam loopback (no radio needed)
#   ./run_grc.sh examples/foo.grc       # a specific flowgraph
#
# The Windows equivalent is RUN_GRC_WINDOWS.bat.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLOWGRAPH="${1:-$PROJECT_ROOT/examples/aj_command_loopback.grc}"

if ! command -v gnuradio-companion >/dev/null 2>&1; then
    echo "ERROR: gnuradio-companion is not on PATH." >&2
    echo "Activate your GNU Radio environment (conda activate, or install gnuradio)." >&2
    exit 1
fi

# Both are needed: PYTHONPATH so the generated flowgraph can import gpsk_comms,
# GRC_BLOCKS_PATH so Companion can find the block definitions in grc/.
export PYTHONPATH="$PROJECT_ROOT/python:${PYTHONPATH:-}"
export GRC_BLOCKS_PATH="$PROJECT_ROOT/grc:${GRC_BLOCKS_PATH:-}"

if ! python3 -c "from gpsk_comms import aj_command_tx, gmsk_command_tx" 2>/dev/null; then
    echo "ERROR: the gpsk_comms blocks could not be imported." >&2
    echo "Check that $PROJECT_ROOT/python/gpsk_comms exists and numpy is available." >&2
    exit 1
fi

echo "Project:   $PROJECT_ROOT"
echo "Flowgraph: $FLOWGRAPH"
if [ -z "${GPSK_COMMS_KEY_FILE:-}${GPSK_COMMS_KEY:-}" ]; then
    echo
    echo "Note: GPSK_COMMS_KEY_FILE is not set. The loopback example generates its"
    echo "own ephemeral key, but any real link needs a shared key:"
    echo "    python3 -m gpsk_comms.security --output link.key"
    echo "    export GPSK_COMMS_KEY_FILE=\$PWD/link.key"
fi
echo

exec gnuradio-companion "$FLOWGRAPH"
