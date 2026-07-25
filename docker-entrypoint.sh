#!/bin/bash
set -e

# Activate the uv-managed venv so every python/pip call uses it.
# The venv is created by the Dockerfile at /workspace/LeggedGym-Ex/.venv.
. /workspace/LeggedGym-Ex/.venv/bin/activate

# If manual arguments are passed (e.g. `docker run ... bash` or
# `docker run ... python ...`), execute them directly and skip the
# automatic swap_experiment launch.
if [ $# -gt 0 ]; then
    exec "$@"
fi

# ---------------------------------------------------------------------------
# Automatic launch of swap_experiment.py
# ---------------------------------------------------------------------------
# Policies are auto-discovered from /workspace/policies/*.pt.
# Each .pt file is registered as --policy <basename_without_ext>:<path>.
#
# Configure via environment variables in docker-compose.yml or a .env file:
#
#   ACTIVE_POLICY=<name>   (must match a discovered .pt filename without ext)
#   CONTROL_PORT=<int>     (default: 9013)
#   VISER_PORT=<int>       (default: 9006)
#   HEADLESS=1|0           (default: 0)
#   SPEED=<float>          (default: 0.35)
# ---------------------------------------------------------------------------

export GENESIS_BACKEND=${GENESIS_BACKEND:-cpu}

ARGS=()

# ---------------------------------------------------------------------------
# Auto-discover policies
# ---------------------------------------------------------------------------
POLICIES_DIR="/workspace/policies"
POLICY_NAMES=()

if [ -d "$POLICIES_DIR" ]; then
    for pt_file in "$POLICIES_DIR"/*.pt; do
        [ -f "$pt_file" ] || continue
        name=$(basename "$pt_file" .pt)
        POLICY_NAMES+=("$name")
        ARGS+=("--policy" "${name}:${pt_file}")
    done
fi

if [ "${#POLICY_NAMES[@]}" -eq 0 ]; then
    echo "ERROR: No policies found in ${POLICIES_DIR}/"
    echo "Mount a policies/ directory containing one or more .pt checkpoint files."
    echo "Each file will be registered as a policy named after its filename (without .pt)."
    exit 1
fi

echo "Discovered ${#POLICY_NAMES[@]} policy(ies): ${POLICY_NAMES[*]}"

# ---------------------------------------------------------------------------
# Validate and set active policy
# ---------------------------------------------------------------------------
active_name=""

if [ -n "$ACTIVE_POLICY" ]; then
    # Validate that ACTIVE_POLICY matches a discovered name
    found=0
    for n in "${POLICY_NAMES[@]}"; do
        if [ "$n" = "$ACTIVE_POLICY" ]; then
            found=1
            break
        fi
    done
    if [ "$found" -eq 0 ]; then
        echo "ERROR: ACTIVE_POLICY='${ACTIVE_POLICY}' does not match any discovered policy."
        echo "Available policies: ${POLICY_NAMES[*]}"
        exit 1
    fi
    active_name="$ACTIVE_POLICY"
else
    # Default to the first policy alphabetically
    active_name="${POLICY_NAMES[0]}"
    echo "ACTIVE_POLICY not set; defaulting to '${active_name}'."
fi

ARGS+=("--active" "$active_name")

# ---------------------------------------------------------------------------
# Optional arguments
# ---------------------------------------------------------------------------
if [ -n "$CONTROL_PORT" ]; then
    ARGS+=("--control_port" "$CONTROL_PORT")
else
    ARGS+=("--control_port" "9013")
fi

if [ -n "$VISER_PORT" ]; then
    ARGS+=("--viser_port" "$VISER_PORT")
else
    ARGS+=("--viser_port" "9006")
fi

if [ "$HEADLESS" = "1" ] || [ "$HEADLESS" = "true" ]; then
    ARGS+=("--headless")
fi

if [ -n "$SPEED" ]; then
    ARGS+=("--speed" "$SPEED")
fi

cd /workspace/LeggedGym-Ex
exec python legged_gym/scripts/swap_experiment.py "${ARGS[@]}"
