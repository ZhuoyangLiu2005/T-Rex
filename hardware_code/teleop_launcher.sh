#!/bin/bash
# ============================================================================
# Teleoperation Launcher Script
# Opens 5 tabs in a single terminal window, each running a specific command
# for the robot teleoperation pipeline.
#
# After each command exits (crash or Ctrl-C), the tab drops you into an
# interactive shell that is ALREADY in the correct directory (or ssh session)
# with the python command pre-loaded into history — press Up Arrow + Enter
# to relaunch.
#
# USAGE:
#   Run this script from INSIDE an already-open terminal window of the same
#   emulator (gnome-terminal, konsole, xfce4-terminal, or tilix). The new
#   tabs will be added to that window.
# ============================================================================

# --- Configuration ---
DELAY_BETWEEN_TERMINALS=5   # seconds to wait between launching tabs

# Repo root = directory containing this script.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Conda environment used by the local tabs (see setup.sh / README).
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sharpa-dexmate-tmp}"

# Sharpa Manus SDK checkout (glove client + retargeting), see third_party/README.md.
MANUS_SDK_DIR="${MANUS_SDK_DIR:-$PROJECT_ROOT/third_party/sharpa-manus-sdk}"
RETARGETING_DIR="$(ls -d "$MANUS_SDK_DIR"/retargeting_alg_release_* 2>/dev/null | sort | tail -1)"
if [ -z "$RETARGETING_DIR" ]; then
    echo "WARNING: no retargeting_alg_release_* dir found under $MANUS_SDK_DIR"
    echo "         (clone https://github.com/sharpa-robotics/sharpa-manus-sdk there, or set MANUS_SDK_DIR)"
fi

# Site-specific camera streamer hosts (tabs 1-2) — override for your setup.
HEAD_CAM_SSH_TARGET="${HEAD_CAM_SSH_TARGET:-dexmate}"
WRIST_CAM_SSH_TARGET="${WRIST_CAM_SSH_TARGET:-user@192.168.50.25}"

# --- Ask for DATA_DIR (used in terminal 5) ---
read -rp "Enter DATA_DIR for main_teleop.py (terminal 5): " DATA_DIR
if [ -z "$DATA_DIR" ]; then
    echo "ERROR: DATA_DIR cannot be empty."
    exit 1
fi

# --- Detect available terminal emulator ---
if command -v gnome-terminal >/dev/null 2>&1; then
    TERM_CMD="gnome-terminal"
elif command -v konsole >/dev/null 2>&1; then
    TERM_CMD="konsole"
elif command -v xfce4-terminal >/dev/null 2>&1; then
    TERM_CMD="xfce4-terminal"
elif command -v tilix >/dev/null 2>&1; then
    TERM_CMD="tilix"
else
    echo "ERROR: No supported tabbed terminal emulator found."
    echo "       (tried: gnome-terminal, konsole, xfce4-terminal, tilix)"
    exit 1
fi
echo "Using terminal emulator: $TERM_CMD"

# ---------------------------------------------------------------------------
# open_local_tab: for tabs 3-5 (local conda env + cd + python)
#
# Args:
#   $1 = tab title
#   $2 = target directory (will cd here, and interactive shell starts here)
#   $3 = python command to run and also pre-seed into history
# ---------------------------------------------------------------------------
open_local_tab() {
    local title="$1"
    local target_dir="$2"
    local py_cmd="$3"

    # Payload executed inside the new tab:
    #   1. Print header
    #   2. Source conda + activate env
    #   3. cd into target dir
    #   4. Run the python command
    #   5. When it exits, seed bash history with the same command and drop into
    #      an interactive shell in the SAME directory with conda still active.
    #      `--rcfile` lets us inject `history -s "$py_cmd"` so Up Arrow works
    #      immediately — we write a tiny rc file that first sources the normal
    #      ~/.bashrc, then adds the command to history.
    local payload
    payload=$(cat <<EOF
echo "=============================================================="
echo " TAB: $title"
echo "--------------------------------------------------------------"
echo " DIR:     $target_dir"
echo " COMMAND: $py_cmd"
echo "=============================================================="
echo
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then source "$PROJECT_ROOT/.venv/bin/activate"; else source "\$(conda info --base)/etc/profile.d/conda.sh" && conda activate $CONDA_ENV_NAME; fi
cd "$target_dir" || { echo "cd failed"; exec bash; }
$py_cmd
echo
echo "[command exited — Up Arrow to relaunch, Ctrl-D to close tab]"
# Build a temporary rcfile that preserves env, seeds history, and self-deletes.
RCFILE=\$(mktemp)
cat > "\$RCFILE" <<RCEOF
[ -f ~/.bashrc ] && source ~/.bashrc
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then source "$PROJECT_ROOT/.venv/bin/activate"; else source "\\\$(conda info --base)/etc/profile.d/conda.sh" && conda activate $CONDA_ENV_NAME; fi
cd "$target_dir"
history -s '$py_cmd'
rm -f "\$RCFILE"
RCEOF
exec bash --rcfile "\$RCFILE" -i
EOF
)

    echo ">>> [$title] opening new tab"
    launch_tab "$title" "$payload"
}

# ---------------------------------------------------------------------------
# open_ssh_tab: for tabs 1-2 (remote ssh session)
#
# Args:
#   $1 = tab title
#   $2 = ssh target (e.g. "dexmate" or "user@192.168.50.25")
#   $3 = remote command to run (e.g. "cd repos/... && python foo.py")
#   $4 = remote directory to cd into after the command exits
#   $5 = python-only part of the command, to seed into remote history
# ---------------------------------------------------------------------------
open_ssh_tab() {
    local title="$1"
    local ssh_target="$2"
    local remote_cmd="$3"
    local remote_dir="$4"
    local remote_py="$5"

    # Strategy: use a single ssh -tt session that runs bash -li, which first
    # executes the full pipeline, then — if it exits — seeds history with the
    # python command and hands you an interactive remote shell in the right
    # directory. This way killing python keeps you on the remote host, not
    # back on the local machine.
    #
    # Remote payload (runs on the server under bash -li):
    local remote_payload
    remote_payload="echo '>>> running: $remote_cmd'; $remote_cmd; echo; echo '[remote command exited — Up Arrow to relaunch]'; cd '$remote_dir' 2>/dev/null; history -s '$remote_py'; exec bash -i"

    # Local payload (what the tab runs):
    local payload
    payload=$(cat <<EOF
echo "=============================================================="
echo " TAB: $title (SSH -> $ssh_target)"
echo "--------------------------------------------------------------"
echo " COMMAND: $remote_cmd"
echo "=============================================================="
echo
ssh -tt $ssh_target "bash -lic \"$remote_payload\""
echo
echo "[ssh session closed — Ctrl-D to close tab, or re-run ssh manually]"
exec bash
EOF
)

    echo ">>> [$title] opening new tab"
    launch_tab "$title" "$payload"
}

# ---------------------------------------------------------------------------
# launch_tab: emulator-specific tab launcher
# ---------------------------------------------------------------------------
launch_tab() {
    local title="$1"
    local payload="$2"
    case "$TERM_CMD" in
        gnome-terminal)
            gnome-terminal --tab --title="$title" -- bash -c "$payload"
            ;;
        konsole)
            konsole --new-tab -p "tabtitle=$title" -e bash -c "$payload" &
            ;;
        xfce4-terminal)
            xfce4-terminal --tab --title="$title" -e "bash -c '$payload'"
            ;;
        tilix)
            tilix --action=app-new-session -t "$title" -e "bash -c '$payload'" &
            ;;
    esac
}

# ----------------------------------------------------------------------------
# Tab 1: ssh dexmate -> cam_server stream_sender
# ----------------------------------------------------------------------------
open_ssh_tab \
    "1-dexmate-cam" \
    "$HEAD_CAM_SSH_TARGET" \
    "cd repos/cam_server && python stream_sender.py" \
    "repos/cam_server" \
    "python stream_sender.py"
sleep "$DELAY_BETWEEN_TERMINALS"

# ----------------------------------------------------------------------------
# Tab 2: ssh user@192.168.50.25 -> caip stream_sender
# ----------------------------------------------------------------------------
open_ssh_tab \
    "2-caip-stream" \
    "$WRIST_CAM_SSH_TARGET" \
    "cd Projects/caip && source venv/bin/activate && python stream_sender.py" \
    "Projects/caip" \
    "python stream_sender.py"
sleep "$DELAY_BETWEEN_TERMINALS"

# ----------------------------------------------------------------------------
# Tab 3: SharpaManusClient
# ----------------------------------------------------------------------------
open_local_tab \
    "3-manus-client" \
    "$MANUS_SDK_DIR/client" \
    "./SharpaManusClient.out"
sleep "$DELAY_BETWEEN_TERMINALS"

# ----------------------------------------------------------------------------
# Tab 4: retargeting demo
# ----------------------------------------------------------------------------
open_local_tab \
    "4-retargeting" \
    "$RETARGETING_DIR" \
    "python retargeting_manus_demo_multiprocess.py"
sleep "$DELAY_BETWEEN_TERMINALS"

# ----------------------------------------------------------------------------
# Tab 5: main teleop with user-supplied DATA_DIR
# (the teleop target source — Vive + retargeting — runs in-process)
# ----------------------------------------------------------------------------
open_local_tab \
    "5-main-teleop" \
    "$PROJECT_ROOT/teleop" \
    "python main_teleop.py --data-dir $DATA_DIR"

echo ""
echo "All 5 tabs launched. DATA_DIR=$DATA_DIR"
