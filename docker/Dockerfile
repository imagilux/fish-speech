# docker/Dockerfile — API server and dev images (ROCm GPU + PyTorch + models).
# For the lightweight WebUI image, see docker/Dockerfile.webui.
#
# IMPORTANT: Images do not contain checkpoints. Mount them at runtime.
#
# Build the server:
#   docker build -f docker/Dockerfile --build-arg BACKEND=rocm --target server -t fish-speech-server:rocm .
#
# Build the dev image:
#   docker build -f docker/Dockerfile --build-arg BACKEND=rocm --target dev -t fish-speech-dev:rocm .
#
# Run the server (ROCm):
#   docker run --device /dev/kfd --device /dev/dri -v ./checkpoints:/app/checkpoints -p 8080:8080 fish-speech-server:rocm
#
# Run the server (CPU):
#   docker run -v ./checkpoints:/app/checkpoints -p 8080:8080 fish-speech-server:cpu

ARG UV_EXTRA=rocm72
ARG BACKEND=rocm
ARG ROCM_VER=7.2.1

ARG UBUNTU_VER=24.04
ARG PY_VER=3.12
ARG UV_VERSION=0.8.15

# Create non-root user early for security
ARG USERNAME=fish
ARG USER_UID=1000
ARG USER_GID=1000

##############################################################
# Base stage per backend
##############################################################

# --- ROCm (AMD GPU, x86_64) ---
FROM rocm/dev-ubuntu-${UBUNTU_VER}:${ROCM_VER} AS base-rocm
ENV DEBIAN_FRONTEND=noninteractive
ENV UV_EXTRA=rocm72

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -eux \
    && rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' >/etc/apt/apt.conf.d/keep-cache \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-dev \
        git \
        ca-certificates \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- CPU-only (portable x86_64) ---
FROM python:${PY_VER}-slim AS base-cpu
ENV UV_EXTRA=cpu

# Install system dependencies in a single layer with cleanup
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -eux \
    && rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' >/etc/apt/apt.conf.d/keep-cache \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


##############################################################
# UV stage
##############################################################

ARG UV_VERSION
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-bin

##############################################################
# Shared app base stage
##############################################################

FROM base-${BACKEND} AS app-base

ARG PY_VER
ARG BACKEND
ARG USERNAME
ARG USER_UID
ARG USER_GID
ARG UV_VERSION
ARG UV_EXTRA

ENV BACKEND=${BACKEND} \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies for audio processing
ARG DEPENDENCIES=" \
    libsox-dev \
    build-essential \
    cmake \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg"

# Install system dependencies with caching and cleanup
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -eux \
    && rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' >/etc/apt/apt.conf.d/keep-cache \
    && apt-get update \
    && apt-get install -y --no-install-recommends ${DEPENDENCIES} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install specific uv version
COPY --from=uv-bin /uv /uvx /bin/

# RUN groupadd --gid ${USER_GID} ${USERNAME} \
#     && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
#     && mkdir -p /app /home/${USERNAME}/.cache \
#     && chown -R ${USERNAME}:${USERNAME} /app /home/${USERNAME}/.cache

# Create non-root user (or use existing user)
RUN set -eux; \
    if getent group ${USER_GID} >/dev/null 2>&1; then \
        echo "Group ${USER_GID} already exists"; \
    else \
        groupadd -g ${USER_GID} ${USERNAME}; \
    fi; \
    if id -u ${USER_UID} >/dev/null 2>&1; then \
        echo "User ${USER_UID} already exists, using existing user"; \
        EXISTING_USER=$(id -un ${USER_UID}); \
        mkdir -p /app /home/${EXISTING_USER}/.cache; \
        chown -R ${USER_UID}:${USER_GID} /app /home/${EXISTING_USER}/.cache; \
    else \
        useradd -m -u ${USER_UID} -g ${USER_GID} ${USERNAME}; \
        mkdir -p /app /home/${USERNAME}/.cache; \
        chown -R ${USERNAME}:${USERNAME} /app /home/${USERNAME}/.cache; \
    fi

# Create references and cache directories with proper permissions for the non-root user
RUN mkdir -p /app/references /app/.cache/miopen /app/.config/miopen /app/.triton \
    && chown -R ${USER_UID}:${USER_GID} /app/references /app/.cache/miopen /app/.config/miopen /app/.triton \
    && chmod 755 /app/references /app/.cache/miopen /app/.config/miopen /app/.triton

# Set working directory
WORKDIR /app

# Copy dependency files first for better caching
COPY --chown=${USER_UID}:${USER_GID} pyproject.toml uv.lock README.md ./

# Switch to non-root user for package installation
USER ${USER_UID}:${USER_GID}

# Install Python dependencies (cacheable by lockfiles)
# Use a generic cache path that works regardless of username
RUN --mount=type=cache,target=/tmp/uv-cache,uid=${USER_UID},gid=${USER_GID} \
    uv python pin ${PY_VER} \
    && uv sync --extra ${UV_EXTRA} --frozen --no-install-project

# Copy application code
COPY --chown=${USER_UID}:${USER_GID} . .

# Install the local package after copying source code
RUN uv sync --extra ${UV_EXTRA} --frozen

# miopen-conv-fix: bypass PyTorch's workspace=0 bug for large conv layers on ROCm.
# PyTorch 2.11 fixed the bug for small workspace sizes, but the DAC decoder's late
# conv layers (spatial dims up to 1M) need >1 GB workspace and still fall back to
# ConvDirectNaive. This extension uses MIOpen's Immediate Mode API with per-solution
# validation to select fast GEMM kernels safely. No-op on CUDA/CPU.
# See: pytorch/pytorch#150168, github.com/Imagilux/miopen-conv-fix
USER root
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    if [ "${BACKEND}" = "rocm" ]; then \
        apt-get update && apt-get install -y --no-install-recommends rocthrust-dev rocprim-dev miopen-hip-dev; \
    fi
USER ${USER_UID}:${USER_GID}
RUN if [ "${BACKEND}" = "rocm" ]; then \
        git clone --depth 1 https://github.com/Imagilux/miopen-conv-fix.git /tmp/miopen-conv-fix \
        && cd /tmp/miopen-conv-fix \
        && /app/.venv/bin/python setup.py build_ext --inplace \
        && cp -r miopen_conv_fix/ /app/.venv/lib/python${PY_VER}/site-packages/miopen_conv_fix/ \
        && rm -rf /tmp/miopen-conv-fix; \
    fi

# Create common entrypoint script
RUN printf '%s\n' \
    '#!/bin/bash' \
    'set -euo pipefail' \
    '' \
    '# Set user info from build args' \
    'USER_UID='${USER_UID} \
    'USER_GID='${USER_GID} \
    '' \
    '# Logging function' \
    'log() { echo "[$(date +"%Y-%m-%d %H:%M:%S")] $*" >&2; }' \
    '' \
    '# Validate environment' \
    'validate_env() {' \
    '    if [ ! -d "/app/checkpoints" ]; then' \
    '        log "WARNING: /app/checkpoints directory not found. Please mount your checkpoints."' \
    '    fi' \
    '    if [ ! -d "/app/references" ]; then' \
    '        log "WARNING: /app/references directory not found. Please mount your references."' \
    '    else' \
    '        # Check if we can write to references directory' \
    '        if [ ! -w "/app/references" ]; then' \
    '            log "ERROR: Cannot write to /app/references directory. Please ensure the mounted directory has proper permissions for user with UID ${USER_UID}."' \
    '            log "You can fix this by running: sudo chown -R ${USER_UID}:${USER_GID} /path/to/your/references"' \
    '            exit 1' \
    '        fi' \
    '    fi' \
    '}' \
    '' \
    '# Build device arguments' \
    'build_device_args() {' \
    '    if [ "${BACKEND:-}" = "cpu" ]; then' \
    '        echo "--device cpu"' \
    '    fi' \
    '}' \
    '' \
    '# Build compile arguments' \
    'build_compile_args() {' \
    '    if [ "${1:-}" = "compile" ] || [ "${COMPILE:-}" = "1" ] || [ "${COMPILE:-}" = "true" ]; then' \
    '        echo "--compile"' \
    '        shift' \
    '    fi' \
    '    echo "$@"' \
    '}' \
    '' \
    '# Health check function' \
    'health_check() {' \
    '    local port=${1:-7860}' \
    '    local endpoint=${2:-/health}' \
    '    curl -f http://localhost:${port}${endpoint} 2>/dev/null || exit 1' \
    '}' \
    > /app/common.sh && chmod +x /app/common.sh

##############################################################
# App stages
##############################################################

# API Server
FROM app-base AS server
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

ARG API_SERVER_NAME="0.0.0.0"
ARG API_SERVER_PORT=8080
ARG LLAMA_CHECKPOINT_PATH="checkpoints/s2-pro"
ARG DECODER_CHECKPOINT_PATH="checkpoints/s2-pro/codec.pth"
ARG DECODER_CONFIG_NAME="modded_dac_vq"

# Expose port
EXPOSE ${API_SERVER_PORT}

# Set environment variables
ENV API_SERVER_NAME=${API_SERVER_NAME}
ENV API_SERVER_PORT=${API_SERVER_PORT}
ENV LLAMA_CHECKPOINT_PATH=${LLAMA_CHECKPOINT_PATH}
ENV DECODER_CHECKPOINT_PATH=${DECODER_CHECKPOINT_PATH}
ENV DECODER_CONFIG_NAME=${DECODER_CONFIG_NAME}

# Create server entrypoint
RUN printf '%s\n' \
    '#!/bin/bash' \
    'source /app/common.sh' \
    '' \
    'log "Starting Fish Speech API Server..."' \
    'validate_env' \
    '' \
    'DEVICE_ARGS=$(build_device_args)' \
    'COMPILE_ARGS=$(build_compile_args "$@")' \
    '' \
    'log "Device args: ${DEVICE_ARGS:-none}"' \
    'log "Compile args: ${COMPILE_ARGS}"' \
    'log "Server: ${API_SERVER_NAME}:${API_SERVER_PORT}"' \
    '' \
    'exec uv run --no-sync tools/api_server.py \' \
    '  --listen "${API_SERVER_NAME}:${API_SERVER_PORT}" \' \
    '  --llama-checkpoint-path "${LLAMA_CHECKPOINT_PATH}" \' \
    '  --decoder-checkpoint-path "${DECODER_CHECKPOINT_PATH}" \' \
    '  --decoder-config-name "${DECODER_CONFIG_NAME}" \' \
    '  ${DEVICE_ARGS} ${COMPILE_ARGS}' \
    > /app/start_server.sh && chmod +x /app/start_server.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${API_SERVER_PORT}/v1/health || exit 1

ENTRYPOINT ["/app/start_server.sh"]

# Development stage
FROM app-base AS dev
USER root

# Install development tools
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        vim \
        htop \
        strace \
        gdb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER ${USER_UID}:${USER_GID}

# Install development dependencies
RUN uv sync --extra ${UV_EXTRA} --dev

# Default to bash for development
ENTRYPOINT ["/bin/bash"]
