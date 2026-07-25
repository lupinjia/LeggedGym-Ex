# =============================================================================
# LeggedGym-Ex Dockerfile (Genesis — swap_experiment.py)
# =============================================================================
# This Dockerfile builds a self-contained image for the GIAR fork of
# LeggedGym-Ex, targeting the policy-switching demo in simulation.
#
# It is configured for:
#   * Python 3.12
#   * Genesis simulator  (CPU fallback by default, GPU libraries included)
#   * viser web viewer (port 9006)
#   * FastAPI unified control web / WebSocket bridge (port 9013)
#
# The container is GPU-ready: if you run it with the NVIDIA Container
# Runtime (--gpus all) Genesis and PyTorch will see the CUDA device.
# If no GPU is available the same image still runs because
# swap_experiment.py initialises Genesis with backend=gs.cpu.
#
# Build:
#   docker build -t leggedgym-ex:genesis .
#
# Run interactively:
#   docker run --rm -it -p 9006:9006 -p 9013:9013 \
#        -v $(pwd)/policies:/workspace/policies:ro \
#        leggedgym-ex:genesis bash
#
# =============================================================================

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# Prevent apt-get from asking questions
ENV DEBIAN_FRONTEND=noninteractive

# The legged_gym import machinery refuses to load unless this is set
ENV SIMULATOR=genesis

# uv HTTP timeout (some packages are large)
ENV UV_HTTP_TIMEOUT=300

# ---- Install uv (Rust-based Python package manager) --------------------------
# uv installs its own managed Python versions, creates venvs, and installs
# wheels without ever touching the system pip or distutils.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ---- Install system dependencies ---------------------------------------------
# Graphics libraries (libgl, libegl) and libgomp are required by Genesis,
# viser and Pygame.  build-essential is needed for compiling any source
# distributions that lack pre-built wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    libegl1 \
    libgl1-mesa-glx \
    libglu1-mesa \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Python 3.12 via uv ----------------------------------------------
RUN uv python install 3.12

# ---- Working directory -------------------------------------------------------
WORKDIR /workspace/LeggedGym-Ex

# ---- Copy dependency descriptor first (helps Docker layer caching) -----------
COPY pyproject.toml ./

# ---- Create venv and install dependencies from pyproject.toml -----------------
# pyproject.toml is the single source of truth:
#   * Base dependencies under [project] dependencies
#   * Genesis-specific extras (pinned PyTorch cu126, genesis-world, warp-lang)
#     under [project.optional-dependencies] genesis
# A fresh resolution is generated on every build so the image always picks up
# latest compatible versions without requiring a committed lockfile.
RUN uv venv --python 3.12 .venv \
 && . .venv/bin/activate \
 && uv pip install -r pyproject.toml --extra genesis \
    --extra-index-url https://download.pytorch.org/whl/cu126

# ---- Copy the full repository ------------------------------------------------
COPY . .

# ---- Install the package itself in editable mode -----------------------------
# --no-deps is safe because all runtime requirements were installed above.
RUN . .venv/bin/activate && uv pip install -e . --no-deps

# Ensure imports work regardless of the current working directory
ENV PYTHONPATH=/workspace/LeggedGym-Ex
ENV PATH="/workspace/LeggedGym-Ex/.venv/bin:$PATH"

# ---- Copy the automatic entrypoint ------------------------------------------
COPY docker-entrypoint.sh /workspace/LeggedGym-Ex/docker-entrypoint.sh

# ---- Expose ports used by swap_experiment.py ---------------------------------
# 9006 -> viser web viewer
# 9013 -> unified control web (when --control_port 9013 is passed)
EXPOSE 9006 9013

# Invoking via bash avoids any host-side permission issues with the copied
# script's execute bit.
ENTRYPOINT ["bash", "/workspace/LeggedGym-Ex/docker-entrypoint.sh"]

# When no explicit command is given, the entrypoint launches swap_experiment.py.
# You can still drop into a shell with: docker run ... bash
CMD []
