#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PET_NODE_RANK="${PET_NODE_RANK:?The platform must provide PET_NODE_RANK.}"
PET_NPROC_PER_NODE="${PET_NPROC_PER_NODE:?The platform must provide PET_NPROC_PER_NODE.}"
ENV_DIR="${CHUANGZHI_VENV_DIR:-$ROOT/.venv}"
REQUIREMENTS_FILE="${CHUANGZHI_REQUIREMENTS:-$ROOT/requirements-chuangzhi.txt}"
READY_FILE="$ENV_DIR/chuangzhi_environment_ready.json"
LOCK_DIR="$ROOT/.chuangzhi_uv_setup.lock"
raw_job_id="${PET_MASTER_ADDR:-unknown}_${PET_MASTER_PORT:-unknown}"
job_id="$(printf '%s' "$raw_job_id" | tr -c 'A-Za-z0-9._-' '_')"
FAILED_FILE="$ROOT/.chuangzhi_environment_failed_${job_id}.txt"
WAIT_SECONDS="${CHUANGZHI_ENV_WAIT_SECONDS:-1800}"
EXPECTED_CUDA="${CHUANGZHI_EXPECTED_CUDA:-12.8}"
PYTHON_VERSION="${CHUANGZHI_PYTHON_VERSION:-3.12.12}"
TORCH_VERSION="${CHUANGZHI_TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${CHUANGZHI_TORCHVISION_VERSION:-0.22.1}"
PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ROOT/.tools/python}"
PREPARE_ONLY="${CHUANGZHI_ENV_PREPARE_ONLY:-false}"
UV_BIN="${UV_BIN:-}"

if [[ -z "$UV_BIN" && -x "$ROOT/.tools/uv" ]]; then
  UV_BIN="$ROOT/.tools/uv"
fi
if [[ -z "$UV_BIN" ]]; then
  UV_BIN="$(command -v uv || true)"
fi
if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  echo "uv is unavailable. Install the pinned binary at $ROOT/.tools/uv or set UV_BIN." >&2
  exit 1
fi
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
  echo "Missing Chuangzhi requirements: $REQUIREMENTS_FILE" >&2
  exit 1
fi
if ! [[ "$PET_NODE_RANK" =~ ^[0-9]+$ ]] || ! [[ "$PET_NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "PET_NODE_RANK and PET_NPROC_PER_NODE must be valid integers." >&2
  exit 2
fi
case "$PREPARE_ONLY" in
  true|false) ;;
  *) echo "CHUANGZHI_ENV_PREPARE_ONLY must be true or false." >&2; exit 2 ;;
esac

requirements_sha256="$(sha256sum "$REQUIREMENTS_FILE" | awk '{print $1}')"
ready_matches() {
  [[ -x "$ENV_DIR/bin/python" && -f "$READY_FILE" ]] \
    && grep -q "\"requirements_sha256\": \"$requirements_sha256\"" "$READY_FILE" \
    && grep -q "\"expected_cuda\": \"$EXPECTED_CUDA\"" "$READY_FILE" \
    && grep -q "\"requested_python\": \"$PYTHON_VERSION\"" "$READY_FILE" \
    && grep -q "\"requested_torch\": \"$TORCH_VERSION\"" "$READY_FILE" \
    && grep -q "\"requested_torchvision\": \"$TORCHVISION_VERSION\"" "$READY_FILE"
}

if [[ "$PET_NODE_RANK" == "0" ]]; then
  rm -f "$FAILED_FILE"
  lock_acquired=false
  setup_failed() {
    status=$?
    if [[ "$lock_acquired" == "true" ]]; then
      rmdir "$LOCK_DIR" 2>/dev/null || true
    fi
    if (( status != 0 )); then
      printf 'rank 0 environment setup failed at %s with exit code %s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" > "$FAILED_FILE"
    fi
  }
  trap setup_failed EXIT
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] checking shared uv environment at $ENV_DIR"
  if ! ready_matches; then
    deadline=$((SECONDS + WAIT_SECONDS))
    while [[ "$lock_acquired" == "false" ]] && ! ready_matches; do
      if mkdir "$LOCK_DIR" 2>/dev/null; then
        lock_acquired=true
        break
      fi
      if (( SECONDS >= deadline )); then
        echo "Timed out waiting for the cross-job uv setup lock: $LOCK_DIR" >&2
        exit 1
      fi
      echo "Another job is preparing the shared uv environment; waiting."
      sleep 5
    done

    # Recheck after acquiring the lock because another job may have completed
    # the same requirements while this job was waiting.
    if [[ "$lock_acquired" == "true" ]] && ! ready_matches; then
      rm -f "$READY_FILE"
      if [[ ! -x "$ENV_DIR/bin/python" ]]; then
        "$UV_BIN" python install \
          --install-dir "$PYTHON_INSTALL_DIR" \
          --no-bin \
          "$PYTHON_VERSION"
        UV_PYTHON_INSTALL_DIR="$PYTHON_INSTALL_DIR" \
          "$UV_BIN" venv \
            --managed-python \
            --python "$PYTHON_VERSION" \
            "$ENV_DIR"
      fi
      export DS_BUILD_OPS="${DS_BUILD_OPS:-0}"
      export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"
      "$UV_BIN" pip install \
        --python "$ENV_DIR/bin/python" \
        --torch-backend cu128 \
        "torch==$TORCH_VERSION" \
        "torchvision==$TORCHVISION_VERSION"
      "$UV_BIN" pip install \
        --python "$ENV_DIR/bin/python" \
        --requirement "$REQUIREMENTS_FILE"

      REQUIREMENTS_SHA256="$requirements_sha256" \
      EXPECTED_CUDA="$EXPECTED_CUDA" \
      REQUESTED_PYTHON="$PYTHON_VERSION" \
      REQUESTED_TORCH="$TORCH_VERSION" \
      REQUESTED_TORCHVISION="$TORCHVISION_VERSION" \
        "$ENV_DIR/bin/python" - "$READY_FILE.tmp" <<'PY'
import importlib
import importlib.metadata
import json
import os
import pathlib
import platform
import sys

import torch

modules = (
    "accelerate",
    "cv2",
    "deepspeed",
    "diffusers",
    "hydra",
    "numpy",
    "omegaconf",
    "timm",
    "tokenizers",
    "torch",
    "torchvision",
    "transformers",
    "zarr",
)
versions = {}
for name in modules:
    module = importlib.import_module(name)
    versions[name] = str(getattr(module, "__version__", "unknown"))
for distribution in ("accelerate", "deepspeed", "tokenizers", "transformers"):
    versions[f"{distribution}_distribution"] = importlib.metadata.version(distribution)

expected_cuda = os.environ["EXPECTED_CUDA"]
if not str(torch.version.cuda).startswith(expected_cuda):
    raise SystemExit(
        f"Expected a CUDA {expected_cuda} PyTorch image, found torch.version.cuda={torch.version.cuda}."
    )

payload = {
    "python": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "requirements_sha256": os.environ["REQUIREMENTS_SHA256"],
    "expected_cuda": expected_cuda,
    "requested_python": os.environ["REQUESTED_PYTHON"],
    "requested_torch": os.environ["REQUESTED_TORCH"],
    "requested_torchvision": os.environ["REQUESTED_TORCHVISION"],
    "torch_cuda_version": torch.version.cuda,
    "versions": versions,
}
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
      mv "$READY_FILE.tmp" "$READY_FILE"
    fi
    if [[ "$lock_acquired" == "true" ]]; then
      rmdir "$LOCK_DIR"
      lock_acquired=false
    fi
  else
    echo "Shared uv environment already matches requirements $requirements_sha256."
  fi
  trap - EXIT
else
  deadline=$((SECONDS + WAIT_SECONDS))
  echo "Waiting up to ${WAIT_SECONDS}s for rank 0 to prepare $ENV_DIR."
  until ready_matches; do
    if [[ -f "$FAILED_FILE" ]]; then
      cat "$FAILED_FILE" >&2
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for rank 0 environment setup." >&2
      exit 1
    fi
    sleep 5
  done
fi

if [[ "$PREPARE_ONLY" == "true" ]]; then
  echo "Shared Chuangzhi uv environment is installed and ready for GPU jobs."
  exit 0
fi

"$ENV_DIR/bin/python" - "$PET_NPROC_PER_NODE" "$EXPECTED_CUDA" <<'PY'
import importlib
import importlib.metadata
import json
import socket
import sys

required_gpus = int(sys.argv[1])
expected_cuda = sys.argv[2]
modules = (
    "accelerate",
    "cv2",
    "deepspeed",
    "diffusers",
    "hydra",
    "timm",
    "tokenizers",
    "torch",
    "transformers",
    "zarr",
)
versions = {}
for name in modules:
    module = importlib.import_module(name)
    versions[name] = str(getattr(module, "__version__", "unknown"))
for distribution in ("accelerate", "deepspeed", "tokenizers", "transformers"):
    versions[f"{distribution}_distribution"] = importlib.metadata.version(distribution)

import torch

visible_gpus = torch.cuda.device_count()
summary = {
    "hostname": socket.gethostname(),
    "python": sys.version,
    "torch_cuda_version": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "visible_gpu_count": visible_gpus,
    "required_gpu_count": required_gpus,
    "expected_cuda": expected_cuda,
    "versions": versions,
}
print(json.dumps(summary, indent=2, sort_keys=True))
if not torch.cuda.is_available():
    raise SystemExit("The selected GPU image does not expose CUDA to the project venv.")
if visible_gpus < required_gpus:
    raise SystemExit(
        f"Expected at least {required_gpus} visible GPUs, found {visible_gpus}."
    )
if not str(torch.version.cuda).startswith(expected_cuda):
    raise SystemExit(
        f"Expected a CUDA {expected_cuda} PyTorch image, found torch.version.cuda={torch.version.cuda}."
    )
PY
