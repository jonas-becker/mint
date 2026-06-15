#!/bin/bash -l
#SBATCH --job-name=exp3_debate
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=256G
# GLM-4.7-Flash BF16 weights need >40GB VRAM with vLLM 0.19; 40GB nodes OOM during load.
# Request 80GB GPUs for GLM-4.7-Flash; 40GB nodes OOM during model load.
#SBATCH --constraint=80gb_vram
#SBATCH --ntasks=1
#SBATCH --nodes=1
# If jobs fail with "cannot create temp file for here-document: No space left on device",
# your /tmp (or Slurm spool) may be full. This script sets TMPDIR early (see init_exp3_tmpdir).
# Site-specific: request node-local scratch if your cluster supports it, e.g. #SBATCH --tmp=200G

set -euo pipefail

# Log effective RAM budget for this step (helps confirm #SBATCH --mem was applied).
log_slurm_memory_budget() {
    echo "============ Slurm / cgroup memory budget ============"
    echo "JobID=${SLURM_JOB_ID:-<unset>}  NodeList=${SLURM_JOB_NODELIST:-<unset>}"
    echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-<unset>} (Slurm: MiB per node unless your site documents otherwise)"
    echo "SLURM_MEM_PER_CPU=${SLURM_MEM_PER_CPU:-<unset>}"
    echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-<unset>}  SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-<unset>}"
    if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
        echo "--- scontrol show job (memory / TRES) ---"
        scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | grep -Ei 'JobId=|MinMem|Mem=|TRES=|AllocTRES=' || true
    fi
    _cg2_rel=""
    if [ -r /proc/self/cgroup ]; then
        _cg2_rel="$(awk -F: '$1==0 {print $3; exit}' /proc/self/cgroup 2>/dev/null || true)"
    fi
    if [ -n "${_cg2_rel}" ] && [ -r "/sys/fs/cgroup${_cg2_rel}/memory.max" ]; then
        echo "cgroup v2 memory.max: $(tr -d '\n' <"/sys/fs/cgroup${_cg2_rel}/memory.max") bytes (max=max means no cgroup hard cap)"
    fi
    _cg1_mem=""
    if [ -r /proc/self/cgroup ]; then
        _cg1_mem="$(awk -F: '$2=="memory" {print $3; exit}' /proc/self/cgroup 2>/dev/null || true)"
    fi
    if [ -n "${_cg1_mem}" ] && [ -r "/sys/fs/cgroup/memory${_cg1_mem}/memory.limit_in_bytes" ]; then
        echo "cgroup v1 memory.limit_in_bytes: $(cat "/sys/fs/cgroup/memory${_cg1_mem}/memory.limit_in_bytes" 2>/dev/null || true)"
    fi
    if command -v free >/dev/null 2>&1; then
        echo "--- free -h (whole node; job may be limited to cgroup above) ---"
        free -h || true
    fi
    echo "======================================================"
}
log_slurm_memory_budget

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
if [ ! -f "${REPO_ROOT}/exp3.py" ]; then
    REPO_ROOT="${SCRIPT_DIR}"
fi

PROJECT_HOME="${PROJECT_HOME:-${HOME}}"
MALLM_REPO="${MALLM_REPO:-$(cd "${REPO_ROOT}/.." && pwd)/mallm}"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
VLLM_IMAGE="${VLLM_IMAGE:-${VLLM_IMAGE:-/path/to/vllm-openai.sif}}"
TGI_IMAGE="${TGI_IMAGE:-}"
TGI_MAX_CONCURRENT_REQUESTS="${TGI_MAX_CONCURRENT_REQUESTS:-280}"
TGI_MAX_TOTAL_TOKENS="${TGI_MAX_TOTAL_TOKENS:-}"
TGI_MAX_BATCH_TOTAL_TOKENS="${TGI_MAX_BATCH_TOTAL_TOKENS:-}"
TGI_SERVER_VENV_PATH="${TGI_SERVER_VENV_PATH:-}"
SGLANG_SERVER_VENV_PATH="${SGLANG_SERVER_VENV_PATH:-}"
SGLANG_TP="${SGLANG_TP:-}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-}"
SGLANG_TOOL_CALL_PARSER="${SGLANG_TOOL_CALL_PARSER:-}"
SGLANG_REASONING_PARSER="${SGLANG_REASONING_PARSER:-}"
EXP3_CONCURRENT_DEBATE_PROBE="${EXP3_CONCURRENT_DEBATE_PROBE:-64}"
MALLM_SMOKE_CONCURRENT="${MALLM_SMOKE_CONCURRENT:-1}"
HF_CACHE_VOLUME="${HF_CACHE_VOLUME:-${PROJECT_HOME}/.cache/huggingface}"
SHARED_HF_CACHE_VOLUME="${SHARED_HF_CACHE_VOLUME:-${SHARED_HF_CACHE_VOLUME:-/shared/hf_cache}}"
VLLM_CACHE_VOLUME="${VLLM_CACHE_VOLUME:-${PROJECT_HOME}/.cache/vllm}"
VENV_PATH="${VENV_PATH:-${PROJECT_HOME}/.venvs/misinformed_agents_exp2}"
VLLM_SERVER_BACKEND="${VLLM_SERVER_BACKEND:-}"
VLLM_SERVER_VENV_PATH="${VLLM_SERVER_VENV_PATH:-${PROJECT_HOME}/.venvs/vllm}"
GLM_PYTHON_OVERRIDES_DIR="${GLM_PYTHON_OVERRIDES_DIR:-${PROJECT_HOME}/glm47_pyoverrides}"
APPTAINER_CACHEDIR_PATH="${APPTAINER_CACHEDIR_PATH:-${PROJECT_HOME}/.cache/apptainer}"
APPTAINER_TMPDIR_PATH="${APPTAINER_TMPDIR_PATH:-${PROJECT_HOME}/.cache/apptainer-tmp}"
JOB_TMP_BASE="${JOB_TMP_BASE:-${PROJECT_HOME}/.cache/job-tmp}"

# shellcheck source=scripts/model_slurm_config.sh
source "${REPO_ROOT}/scripts/model_slurm_config.sh"

init_exp3_tmpdir() {
    local _cand=""
    for _cand in "${SLURM_TMPDIR:-}" "${JOB_TMP_BASE}/slurm_tmp_${SLURM_JOB_ID:-$$}"; do
        if [ -n "${_cand}" ] && mkdir -p "${_cand}" 2>/dev/null; then
            if touch "${_cand}/.exp3_tmp_probe_$$" 2>/dev/null; then
                rm -f "${_cand}/.exp3_tmp_probe_$$" 2>/dev/null || true
                export TMPDIR="${_cand}"
                export TMP="${TMPDIR}"
                export TEMP="${TMPDIR}"
                echo "[INFO] TMPDIR=${TMPDIR} (bash here-docs, pip, etc.)"
                df -h "${TMPDIR}" 2>/dev/null || df -h . 2>/dev/null || true
                return 0
            fi
        fi
    done
    echo "[WARN] Could not set a writable TMPDIR under SLURM_TMPDIR or JOB_TMP_BASE; using system default (may be /tmp)."
}
init_exp3_tmpdir

MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-120b}"
MODEL_NAME="$(normalize_model_name "${MODEL_NAME}")"
echo "Normalized MODEL_NAME=${MODEL_NAME}"
AUTO_FALLBACK_ON_40GB="${AUTO_FALLBACK_ON_40GB:-1}"
MODEL_FALLBACK_FOR_40GB="${MODEL_FALLBACK_FOR_40GB:-zai-org/GLM-4.5-Air-FP8}"
GPU_TOTAL_MEM_MIB_OVERRIDE="${GPU_TOTAL_MEM_MIB_OVERRIDE:-}"
SEED="${SEED:-0}"
DEBUG="${DEBUG:-0}"
VERIFY_ONLY="${VERIFY_ONLY:-0}"
PORT="${PORT:-8080}"
ENDPOINT_URL="${ENDPOINT_URL:-http://127.0.0.1:${PORT}/v1}"
HF_TOKEN="${HF_TOKEN:-}"
VLLM_LOG="${VLLM_LOG:-${REPO_ROOT}/out/apptainer_vllm_exp3_${SLURM_JOB_ID}.log}"
MALLM_SMOKE_OUT="${MALLM_SMOKE_OUT:-${JOB_TMP_BASE}/mallm_smoke_${SLURM_JOB_ID}.json}"
MALLM_SMOKE_IN="${MALLM_SMOKE_IN:-${JOB_TMP_BASE}/mallm_smoke_${SLURM_JOB_ID}.jsonl}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-}"
VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-}"
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-}"
VLLM_ENABLE_AUTO_TOOL_CHOICE="${VLLM_ENABLE_AUTO_TOOL_CHOICE:-}"
VLLM_NO_ENABLE_LOG_REQUESTS="${VLLM_NO_ENABLE_LOG_REQUESTS:-}"
VLLM_NO_ENABLE_PREFIX_CACHING="${VLLM_NO_ENABLE_PREFIX_CACHING:-}"
VLLM_GLM_TRANSFORMERS_SITE="${VLLM_GLM_TRANSFORMERS_SITE:-${HOME}/.cache/glm_vllm_transformers}"
PYTHON_BIN="${PYTHON_BIN:-}"
SETUP_ONLY="${SETUP_ONLY:-0}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
ALLOW_RUNTIME_PIP="${ALLOW_RUNTIME_PIP:-0}"
USE_MOCK="${USE_MOCK:-0}"
ALLOW_DOCKER_PULL="${ALLOW_DOCKER_PULL:-0}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-}"
GLM_OPENAI_SERVER_SCRIPT="${GLM_OPENAI_SERVER_SCRIPT:-${REPO_ROOT}/glm_openai_server.py}"
MALLM_OPENAI_TIMEOUT_SECONDS="${MALLM_OPENAI_TIMEOUT_SECONDS:-3600}"
MALLM_OPENAI_MAX_RETRIES="${MALLM_OPENAI_MAX_RETRIES:-5}"
GLM_SHIM_QUEUE_KEEPALIVE_SECONDS="${GLM_SHIM_QUEUE_KEEPALIVE_SECONDS:-5}"
GLM_SHIM_MAX_QUEUE_WAIT_SECONDS="${GLM_SHIM_MAX_QUEUE_WAIT_SECONDS:-600}"
GLM_SHIM_STREAMER_TIMEOUT_SECONDS="${GLM_SHIM_STREAMER_TIMEOUT_SECONDS:-120}"
MALLM_CONCURRENT_API_REQUESTS="${MALLM_CONCURRENT_API_REQUESTS:-250}"

if [ ! -d "${REPO_ROOT}" ]; then
    echo "Experiment repo does not exist: ${REPO_ROOT}"
    exit 2
fi

if [ ! -d "${MALLM_REPO}" ]; then
    echo "MALLM repo does not exist: ${MALLM_REPO}"
    exit 2
fi

apply_model_hf_cache_path "${MODEL_NAME}"

mkdir -p "${REPO_ROOT}/out"
mkdir -p "${HF_CACHE_VOLUME}" "${HF_CACHE_VOLUME}/hub" "${HF_CACHE_VOLUME}/transformers"
mkdir -p "${JOB_TMP_BASE}"

if [ -n "${SHARED_MODEL_CACHE_SOURCE:-}" ]; then
    if [ ! -d "${SHARED_MODEL_CACHE_SOURCE}" ]; then
        echo "Shared model cache source not found: ${SHARED_MODEL_CACHE_SOURCE}"
        exit 2
    fi
    if [ ! -e "${LOCAL_MODEL_CACHE_TARGET}" ]; then
        ln -s "${SHARED_MODEL_CACHE_SOURCE}" "${LOCAL_MODEL_CACHE_TARGET}"
    fi
fi
mkdir -p "${VLLM_CACHE_VOLUME}"
mkdir -p "${APPTAINER_CACHEDIR_PATH}" "${APPTAINER_TMPDIR_PATH}"

export TMPDIR="${TMPDIR:-${JOB_TMP_BASE}}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"

cd "${REPO_ROOT}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "nvidia-smi not found. This job is expected to run on a GPU partition."
    exit 2
fi

GPU_TOTAL_MEM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1 {print int($1)}')"
if [ -n "${GPU_TOTAL_MEM_MIB_OVERRIDE}" ]; then
    GPU_TOTAL_MEM_MIB="${GPU_TOTAL_MEM_MIB_OVERRIDE}"
fi
if [ -z "${GPU_TOTAL_MEM_MIB}" ]; then
    GPU_TOTAL_MEM_MIB=0
fi
if [ "${AUTO_FALLBACK_ON_40GB}" = "1" ] && [ "${MODEL_NAME}" = "zai-org/GLM-4.7-Flash" ] && [ "${GPU_TOTAL_MEM_MIB}" -gt 0 ] && [ "${GPU_TOTAL_MEM_MIB}" -lt 70000 ]; then
    echo "Detected ${GPU_TOTAL_MEM_MIB} MiB GPU memory (< 70GB)."
    echo "Switching model from ${MODEL_NAME} to ${MODEL_FALLBACK_FOR_40GB} to avoid startup OOM on 40GB GPUs."
    MODEL_NAME="${MODEL_FALLBACK_FOR_40GB}"
fi
apply_model_vllm_max_model_len_default "${MODEL_NAME}"
echo "Using VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN} for model ${MODEL_NAME}"
ensure_hub_model_cached_or_exit "${MODEL_NAME}"
if command -v module >/dev/null 2>&1; then
    module load env/staging/2024.1 >/dev/null 2>&1 || true
    module load Python/3.11.10-GCCcore-13.3.0 >/dev/null 2>&1 || true
    module load Apptainer >/dev/null 2>&1 || true
    module load apptainer >/dev/null 2>&1 || true
    module load Singularity >/dev/null 2>&1 || true
fi

if [ -z "${PYTHON_BIN}" ]; then
    for candidate in python3.11 python3.10 python3.9 python3; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            PYTHON_BIN="${candidate}"
            break
        fi
    done
fi

if [ -z "${PYTHON_BIN}" ]; then
    echo "No usable Python interpreter found."
    exit 2
fi

"${PYTHON_BIN}" --version
if ! "${PYTHON_BIN}" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 9) else 1)
PY
then
    echo "Python >= 3.9 required for MALLM. Found: $("${PYTHON_BIN}" --version 2>&1)"
    echo "Try loading a newer Python module or set PYTHON_BIN explicitly."
    exit 2
fi

if [ -d "${VENV_PATH}" ] && { [ ! -f "${VENV_PATH}/bin/activate" ] || [ ! -x "${VENV_PATH}/bin/python" ]; }; then
    echo "Existing venv is incomplete; recreating: ${VENV_PATH}"
    rm -rf "${VENV_PATH}"
fi

if [ -x "${VENV_PATH}/bin/python" ]; then
    if ! "${VENV_PATH}/bin/python" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 9) else 1)
PY
    then
        echo "Existing venv uses Python <3.9; recreating: ${VENV_PATH}"
        rm -rf "${VENV_PATH}"
    fi
fi

if [ ! -d "${VENV_PATH}" ]; then
    echo "Creating dedicated exp3 environment at ${VENV_PATH}"
    "${PYTHON_BIN}" -m venv "${VENV_PATH}"
fi

echo "Activating environment at ${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
python --version

if [ -n "${TGI_SERVER_VENV_PATH}" ] && [ -x "${TGI_SERVER_VENV_PATH}/bin/text-generation-launcher" ]; then
  export PATH="${TGI_SERVER_VENV_PATH}/bin:${PATH}"
  echo "Prepended TGI_SERVER_VENV_PATH to PATH for text-generation-launcher."
fi

if [ -n "${SGLANG_SERVER_VENV_PATH}" ] && [ -d "${SGLANG_SERVER_VENV_PATH}/bin" ]; then
  export PATH="${SGLANG_SERVER_VENV_PATH}/bin:${PATH}"
  echo "Prepended SGLANG_SERVER_VENV_PATH to PATH."
fi

if [ "${VERIFY_ONLY}" = "1" ]; then
    echo "VERIFY_ONLY=1 -> exp3 Slurm setup looks good."
    echo "Repo root: ${REPO_ROOT}"
    echo "MALLM repo: ${MALLM_REPO}"
    echo "Slurm account: ${SLURM_JOB_ACCOUNT:-unknown}"
    echo "Slurm partition: ${SLURM_JOB_PARTITION:-unknown}"
    python --version
    scontrol show job "${SLURM_JOB_ID}" || true
    exit 0
fi

export HF_HOME="${HF_CACHE_VOLUME}"
export HF_HUB_CACHE="${HF_CACHE_VOLUME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_VOLUME}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE_VOLUME}/transformers"
export MALLM_DISABLE_CONTEXT="1"

if is_hub_model_id "${MODEL_NAME}"; then
    if is_model_cached_in "${MODEL_NAME}" "${HF_CACHE_VOLUME}"; then
        if [ -z "${HF_HUB_OFFLINE}" ]; then
            HF_HUB_OFFLINE="1"
        fi
        if [ -z "${TRANSFORMERS_OFFLINE}" ]; then
            TRANSFORMERS_OFFLINE="1"
        fi
    fi
fi

if [ "${HF_HUB_OFFLINE}" = "1" ] && is_hub_model_id "${MODEL_NAME}"; then
    if ! is_model_cached_in "${MODEL_NAME}" "${HF_CACHE_VOLUME}"; then
        echo "HF_HUB_OFFLINE=1 but model is not cached: ${MODEL_NAME}"
        echo "Cache checked: ${HF_CACHE_VOLUME}/hub/models--${MODEL_NAME//\//--}/snapshots"
        exit 2
    fi
fi

if [ -n "${HF_HUB_OFFLINE}" ]; then
    export HF_HUB_OFFLINE
fi
if [ -n "${TRANSFORMERS_OFFLINE}" ]; then
    export TRANSFORMERS_OFFLINE
fi

if python - <<'PY'
import importlib
modules = [
    "datasets", "requests", "langchain", "langchain_core", "openai", "rich",
    "mallm", "nltk"
]
for name in modules:
    importlib.import_module(name)
print("Runtime dependencies and MALLM import check passed")
PY
then
    echo "Using existing dependencies from ${VENV_PATH}"
else
    if [ "${ALLOW_RUNTIME_PIP}" != "1" ]; then
        echo "Dependencies missing and runtime pip install is disabled (ALLOW_RUNTIME_PIP=0)."
        echo "This cluster's compute nodes may not reach PyPI; prebuild venv on login node or rerun with ALLOW_RUNTIME_PIP=1."
        exit 2
    fi
    echo "Installing runtime dependencies inside job (ALLOW_RUNTIME_PIP=1)"
    python -m pip install --upgrade pip setuptools wheel "poetry-core>=1.9.0"
    python -m pip install \
        datasets requests langchain langchain-core openai rich fire colorama cffi \
        "huggingface-hub==0.23.0" nltk json-repair contextplus langdetect immutabledict
    python -m pip install --no-deps -e "${MALLM_REPO}"
    python - <<'PY'
import importlib
modules = [
    "datasets", "requests", "langchain", "langchain_core", "openai", "rich",
    "mallm", "nltk"
]
for name in modules:
    importlib.import_module(name)
print("Runtime dependencies and MALLM import check passed")
PY
fi

if [ "${SETUP_ONLY}" = "1" ]; then
    echo "SETUP_ONLY=1 -> dependency setup successful. Exiting before vLLM launch."
    exit 0
fi

run_mock_mode() {
    DEBUG_FLAG=""
    if [ "${DEBUG}" = "1" ]; then
        DEBUG_FLAG="--debug"
    fi
    CMD="python exp3.py --mock --model_name ${MODEL_NAME} --seed ${SEED} ${DEBUG_FLAG} --continue"
    echo "Running exp3 in mock mode: ${CMD}"
    eval "${CMD}"
}

if [ "${USE_MOCK}" = "1" ]; then
    run_mock_mode
    exit 0
fi

export PYTHONPATH="${MALLM_REPO}:${PYTHONPATH:-}"
export MALLM_ENDPOINT_URL="${ENDPOINT_URL}"
export OPENAI_BASE_URL="${ENDPOINT_URL}"
export MALLM_OPENAI_TIMEOUT_SECONDS
export MALLM_OPENAI_MAX_RETRIES
export GLM_SHIM_QUEUE_KEEPALIVE_SECONDS
export GLM_SHIM_MAX_QUEUE_WAIT_SECONDS
export GLM_SHIM_STREAMER_TIMEOUT_SECONDS
export MALLM_CONCURRENT_API_REQUESTS

export APPTAINERENV_HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
export APPTAINERENV_HF_HOME="/root/.cache/huggingface"
export APPTAINERENV_HF_HUB_CACHE="/root/.cache/huggingface/hub"
export APPTAINERENV_HUGGINGFACE_HUB_CACHE="/root/.cache/huggingface/hub"
export APPTAINERENV_TRANSFORMERS_CACHE="/root/.cache/huggingface/transformers"
export APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER="0"
export APPTAINERENV_HF_HUB_OFFLINE="${HF_HUB_OFFLINE}"
export APPTAINERENV_TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}"
normalize_cuda_visible_devices() {
    local raw="${1:-}"
    raw="${raw//gpu:/}"
    raw="${raw//GPU:/}"
    raw="${raw// /}"
    raw="${raw//;/,}"
    if [[ -z "${raw}" ]]; then
        return 1
    fi
    if [[ "${raw}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "${raw}"
        return 0
    fi
    return 1
}

CUDA_DEVICES_RAW="${CUDA_VISIBLE_DEVICES:-}"
if [ -z "${CUDA_DEVICES_RAW}" ]; then
    CUDA_DEVICES_RAW="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"
fi

if CUDA_DEVICES_NORM="$(normalize_cuda_visible_devices "${CUDA_DEVICES_RAW}")"; then
    :
else
    CUDA_DEVICES_NORM="0"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES_NORM}"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_DEVICES_NORM}"

export APPTAINERENV_XDG_CACHE_HOME="/root/.cache"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR_PATH}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR_PATH}"

ALLOCATED_GPUS="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-1}}"
if [[ "${ALLOCATED_GPUS}" == *","* ]]; then
    ALLOCATED_GPUS="$(echo "${ALLOCATED_GPUS}" | awk -F',' '{print NF}')"
fi
if ! [[ "${ALLOCATED_GPUS}" =~ ^[0-9]+$ ]] || [ "${ALLOCATED_GPUS}" -lt 1 ]; then
    ALLOCATED_GPUS="1"
fi

ensure_model_gpu_allocation "${MODEL_NAME}"
apply_model_vllm_trust_remote_code_default "${MODEL_NAME}"
apply_model_vllm_gpu_mem_default "${MODEL_NAME}"
apply_model_vllm_glm_gptoss_defaults "${MODEL_NAME}"
apply_gpt_oss_tiktoken_env "${MODEL_NAME}"

if [[ "${MODEL_NAME}" == zai-org/GLM-* ]]; then
  if [ "${VLLM_SERVER_BACKEND:-apptainer}" = "apptainer" ] && [ ! -f "${VLLM_GLM_TRANSFORMERS_SITE}/transformers/__init__.py" ]; then
    echo "WARN: GLM + vLLM Apptainer: install a transformers 5.0 staging tree for glm4_moe_lite (bundled 4.57 is too old), e.g. on a login node with Apptainer:"
    echo "  T=\"\${HOME}/.cache/glm_vllm_transformers\"; rm -rf \"\${T:?}\"/*; mkdir -p \"\${T}\""
    echo "  apptainer exec --bind \"\${T}:/out\" \"\${VLLM_IMAGE}\" pip install --target /out \"transformers==5.0.0\" \"huggingface_hub==1.3.0\" tokenizers safetensors regex requests pyyaml filelock tqdm packaging fsspec"
    echo "  rm -rf \"\${T}/numpy\" \"\${T}/numpy-\"*.dist-info \"\${T}/numpy.libs\" 2>/dev/null || true"
    echo "Then set VLLM_GLM_TRANSFORMERS_SITE=\${T} (default) or leave unset. See https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM.html"
  fi
fi

if [ -z "${VLLM_SERVER_BACKEND}" ]; then
  case "${MODEL_NAME}" in
    zai-org/GLM-*)
      VLLM_SERVER_BACKEND="apptainer"
      ;;
    *)
      VLLM_SERVER_BACKEND="apptainer"
      ;;
  esac
fi

if [ "${VLLM_SERVER_BACKEND}" = "sglang" ]; then
  if [ -n "${SGLANG_SERVER_PYTHON:-}" ] && [ -x "${SGLANG_SERVER_PYTHON}" ]; then
    SGLANG_PY="${SGLANG_SERVER_PYTHON}"
  elif [ -n "${SGLANG_SERVER_VENV_PATH:-}" ] && [ -x "${SGLANG_SERVER_VENV_PATH}/bin/python" ]; then
    SGLANG_PY="${SGLANG_SERVER_VENV_PATH}/bin/python"
  else
    SGLANG_PY="python"
  fi
  export SGLANG_SERVER_PYTHON="${SGLANG_PY}"
  if ! "${SGLANG_PY}" -c "import sglang" 2>/dev/null; then
    if [ "${ALLOW_RUNTIME_PIP}" != "1" ]; then
      echo "ERROR: VLLM_SERVER_BACKEND=sglang but SGLang is not importable with: ${SGLANG_PY}"
      echo "Install SGLang in the job venv, set SGLANG_SERVER_VENV_PATH to a venv with sglang, or set SGLANG_SERVER_PYTHON."
      echo "Alternatively: VLLM_SERVER_BACKEND=hf_openai_shim | tgi | apptainer"
      exit 2
    fi
    echo "Installing SGLang (ALLOW_RUNTIME_PIP=1) with ${SGLANG_PY} -m pip ..."
    "${SGLANG_PY}" -m pip install --upgrade pip
    "${SGLANG_PY}" -m pip install "sglang"
    if ! "${SGLANG_PY}" -c "import sglang" 2>/dev/null; then
      echo "ERROR: sglang import still fails after pip install."
      exit 2
    fi
  fi
fi

if [ "${VLLM_SERVER_BACKEND}" = "tgi" ]; then
  if command -v text-generation-launcher >/dev/null 2>&1; then
    :
  else
    if [ -z "${TGI_IMAGE}" ]; then
      echo "ERROR: VLLM_SERVER_BACKEND=tgi but text-generation-launcher is not in PATH and TGI_IMAGE is empty."
      echo "Install TGI in the job environment, or submit with e.g.:"
      echo "  sbatch --export=ALL,TGI_IMAGE=/path/to/text-generation-inference.sif exp3.slurm"
      exit 2
    fi
  fi
fi

if [ "${VLLM_SERVER_BACKEND}" = "tgi" ]; then
  TGI_ARGS=(
    "--model-id" "${MODEL_NAME}"
    "--port" "${PORT}"
    "--hostname" "127.0.0.1"
    "--num-shard" "${ALLOCATED_GPUS}"
    "--max-concurrent-requests" "${TGI_MAX_CONCURRENT_REQUESTS}"
  )
  if [ -n "${TGI_MAX_TOTAL_TOKENS}" ]; then
    TGI_ARGS+=(--max-total-tokens "${TGI_MAX_TOTAL_TOKENS}")
  fi
  if [ -n "${TGI_MAX_BATCH_TOTAL_TOKENS}" ]; then
    TGI_ARGS+=(--max-batch-total-tokens "${TGI_MAX_BATCH_TOTAL_TOKENS}")
  fi
  if [[ "${MODEL_NAME}" == zai-org/GLM-* ]] || [ "${VLLM_TRUST_REMOTE_CODE}" = "1" ] || [ "${VLLM_TRUST_REMOTE_CODE}" = "true" ]; then
    TGI_ARGS+=(--trust-remote-code)
  fi

  if command -v text-generation-launcher >/dev/null 2>&1; then
    VLLM_CMD=(text-generation-launcher "${TGI_ARGS[@]}")
  else
    if [ ! -e "${TGI_IMAGE}" ]; then
      echo "TGI_IMAGE path does not exist (on this node): ${TGI_IMAGE}"
      exit 2
    fi
    if [[ "${TGI_IMAGE}" == docker://* ]] && [ "${ALLOW_DOCKER_PULL}" != "1" ]; then
      echo "TGI_IMAGE uses docker:// but compute nodes cannot pull from registries here."
      echo "Provide a local .sif path via TGI_IMAGE or set ALLOW_DOCKER_PULL=1."
      exit 2
    fi
    if ! command -v "${APPTAINER_BIN}" >/dev/null 2>&1; then
      if command -v apptainer >/dev/null 2>&1; then
        APPTAINER_BIN="apptainer"
      elif command -v singularity >/dev/null 2>&1; then
        APPTAINER_BIN="singularity"
      else
        echo "Apptainer/Singularity not found. Set APPTAINER_BIN to run TGI from TGI_IMAGE."
        exit 2
      fi
    fi
    APPTAINER_BINDS_TGI=(
      "${HF_CACHE_VOLUME}:/root/.cache/huggingface"
    )
    if [[ "${MODEL_NAME}" == zai-org/GLM-* ]] && [ -d "${GLM_PYTHON_OVERRIDES_DIR}" ]; then
      export APPTAINERENV_PYTHONPATH="/glm47_pyoverrides${APPTAINERENV_PYTHONPATH:+:${APPTAINERENV_PYTHONPATH}}"
      APPTAINER_BINDS_TGI+=("${GLM_PYTHON_OVERRIDES_DIR}:/glm47_pyoverrides")
    fi
    VLLM_CMD=(
      "${APPTAINER_BIN}" run --cleanenv --nv
    )
    for bind_path in "${APPTAINER_BINDS_TGI[@]}"; do
      VLLM_CMD+=(--bind "${bind_path}")
    done
    VLLM_CMD+=("${TGI_IMAGE}" "${TGI_ARGS[@]}")
  fi
elif [ "${VLLM_SERVER_BACKEND}" = "sglang" ]; then
  if [ -z "${SGLANG_TP:-}" ]; then
    SGLANG_TP="${ALLOCATED_GPUS}"
  fi
  if [ -z "${SGLANG_TOOL_CALL_PARSER:-}" ]; then
    case "${MODEL_NAME}" in
      zai-org/GLM-4.5*)
        SGLANG_TOOL_CALL_PARSER="glm45"
        ;;
      zai-org/GLM-4.6*|zai-org/GLM-4.7*|zai-org/GLM-5*)
        SGLANG_TOOL_CALL_PARSER="glm47"
        ;;
    esac
  fi
  VLLM_CMD=(
    "${SGLANG_PY}" -m sglang.launch_server
    --model-path "${MODEL_NAME}"
    --host 127.0.0.1
    --port "${PORT}"
    --tp "${SGLANG_TP}"
  )
  if [[ "${MODEL_NAME}" == zai-org/GLM-* ]] || [ "${VLLM_TRUST_REMOTE_CODE}" = "1" ] || [ "${VLLM_TRUST_REMOTE_CODE}" = "true" ]; then
    VLLM_CMD+=(--trust-remote-code)
  fi
  if [ -n "${SGLANG_MEM_FRACTION_STATIC:-}" ]; then
    VLLM_CMD+=(--mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}")
  fi
  if [ -n "${SGLANG_MAX_RUNNING_REQUESTS:-}" ]; then
    VLLM_CMD+=(--max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}")
  fi
  if [ -n "${SGLANG_CONTEXT_LENGTH:-}" ]; then
    VLLM_CMD+=(--context-length "${SGLANG_CONTEXT_LENGTH}")
  fi
  if [ -n "${SGLANG_CHUNKED_PREFILL_SIZE:-}" ]; then
    VLLM_CMD+=(--chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}")
  fi
  if [ -n "${SGLANG_TOOL_CALL_PARSER:-}" ]; then
    VLLM_CMD+=(--tool-call-parser "${SGLANG_TOOL_CALL_PARSER}")
  fi
  if [ -n "${SGLANG_REASONING_PARSER:-}" ]; then
    VLLM_CMD+=(--reasoning-parser "${SGLANG_REASONING_PARSER}")
  fi
elif [ "${VLLM_SERVER_BACKEND}" = "hf_openai_shim" ]; then
  if [ ! -f "${GLM_OPENAI_SERVER_SCRIPT}" ]; then
    echo "GLM OpenAI shim server script not found: ${GLM_OPENAI_SERVER_SCRIPT}"
    exit 2
  fi
  if python - <<'PY'
import importlib
for name in ("torch", "transformers", "fastapi", "uvicorn", "accelerate"):
    importlib.import_module(name)
print("GLM OpenAI shim dependencies available")
PY
  then
    :
  else
    if [ "${ALLOW_RUNTIME_PIP}" != "1" ]; then
      echo "GLM OpenAI shim dependencies missing (need torch, transformers, accelerate, fastapi, uvicorn)."
      echo "Install them in ${VENV_PATH} or rerun with ALLOW_RUNTIME_PIP=1."
      exit 2
    fi
    echo "Installing GLM OpenAI shim dependencies (ALLOW_RUNTIME_PIP=1)"
    python -m pip install --upgrade fastapi uvicorn accelerate
  fi
  SHIM_LOCAL_FILES_ONLY_FLAG=""
  if [ "${HF_HUB_OFFLINE}" = "1" ] || [ "${TRANSFORMERS_OFFLINE}" = "1" ]; then
    SHIM_LOCAL_FILES_ONLY_FLAG="--local-files-only"
  fi
  VLLM_CMD=(
    python "${GLM_OPENAI_SERVER_SCRIPT}"
    --model "${MODEL_NAME}"
    --host "127.0.0.1"
    --port "${PORT}"
    --trust-remote-code
  )
  if [ -n "${SHIM_LOCAL_FILES_ONLY_FLAG}" ]; then
    VLLM_CMD+=("${SHIM_LOCAL_FILES_ONLY_FLAG}")
  fi
elif [ "${VLLM_SERVER_BACKEND}" = "venv" ]; then
  if [ ! -x "${VLLM_SERVER_VENV_PATH}/bin/python" ]; then
    echo "GLM vLLM environment is missing: ${VLLM_SERVER_VENV_PATH}"
    echo "Either create it or override VLLM_SERVER_BACKEND=apptainer and VLLM_IMAGE."
    exit 2
  fi
  if [[ "${MODEL_NAME}" == zai-org/GLM-* ]]; then
    echo "Validating GLM serving environment at ${VLLM_SERVER_VENV_PATH}"
    if ! "${VLLM_SERVER_VENV_PATH}/bin/python" - <<PY
from importlib.metadata import version
import huggingface_hub
import transformers
transformers_version = version("transformers")
huggingface_hub_version = version("huggingface_hub")
print("transformers", transformers_version)
print("huggingface_hub", huggingface_hub_version)
major = int(transformers_version.split(".", 1)[0])
hub_major = int(huggingface_hub_version.split(".", 1)[0])
if (major < 5 and ".dev" not in transformers_version) or hub_major < 1:
    raise SystemExit(1)
PY
    then
      echo "Updating GLM serving dependencies in ${VLLM_SERVER_VENV_PATH} for ${MODEL_NAME}"
      "${VLLM_SERVER_VENV_PATH}/bin/pip" install --upgrade huggingface_hub
      "${VLLM_SERVER_VENV_PATH}/bin/pip" install --upgrade --no-deps \
        "git+https://github.com/huggingface/transformers.git"
      "${VLLM_SERVER_VENV_PATH}/bin/python" - <<PY
from importlib.metadata import version
print("transformers", version("transformers"))
print("huggingface_hub", version("huggingface_hub"))
PY
    fi
  fi
  VLLM_CMD=(
    "${VLLM_SERVER_VENV_PATH}/bin/python" -m vllm.entrypoints.openai.api_server
    --model "${MODEL_NAME}"
    --port "${PORT}"
    --tensor-parallel-size "${ALLOCATED_GPUS}"
    --max-model-len "${VLLM_MAX_MODEL_LEN}"
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  )
else
  if [ -z "${VLLM_IMAGE}" ]; then
    echo "VLLM_IMAGE is not set."
    echo "Set VLLM_IMAGE to a local readable .sif image path, e.g.:"
    echo "  sbatch --export=ALL,VLLM_IMAGE=/path/to/vllm-openai.sif exp3.slurm"
    exit 2
  fi
  if [ ! -e "${VLLM_IMAGE}" ]; then
    echo "VLLM_IMAGE path does not exist: ${VLLM_IMAGE}"
    exit 2
  fi
  if [[ "${VLLM_IMAGE}" == docker://* ]] && [ "${ALLOW_DOCKER_PULL}" != "1" ]; then
    echo "VLLM_IMAGE uses docker:// but compute nodes cannot pull from registries here."
    echo "Provide a local .sif image path via VLLM_IMAGE or set ALLOW_DOCKER_PULL=1 if network access is available."
    exit 2
  fi
  if ! command -v "${APPTAINER_BIN}" >/dev/null 2>&1; then
    if command -v apptainer >/dev/null 2>&1; then
      APPTAINER_BIN="apptainer"
    elif command -v singularity >/dev/null 2>&1; then
      APPTAINER_BIN="singularity"
    else
      echo "Apptainer/Singularity not found. Set APPTAINER_BIN or load the container runtime module."
      exit 2
    fi
  fi
  APPTAINER_BINDS=(
    "${HF_CACHE_VOLUME}:/root/.cache/huggingface"
    "${VLLM_CACHE_VOLUME}:/root/.cache/vllm"
  )
  if [ -d "${SHARED_HF_CACHE_VOLUME}" ]; then
    APPTAINER_BINDS+=("${SHARED_HF_CACHE_VOLUME}:${SHARED_HF_CACHE_VOLUME}")
  fi
  if [ -n "${GPT_OSS_TIKTOKEN_BIND:-}" ]; then
    APPTAINER_BINDS+=("${GPT_OSS_TIKTOKEN_BIND}")
  fi
  if [[ "${MODEL_NAME}" == zai-org/GLM-* ]] && [ -f "${VLLM_GLM_TRANSFORMERS_SITE}/transformers/__init__.py" ]; then
    APPTAINER_BINDS+=("${VLLM_GLM_TRANSFORMERS_SITE}:/glm_tf_override")
    export APPTAINERENV_PYTHONPATH="/glm_tf_override${APPTAINERENV_PYTHONPATH:+:${APPTAINERENV_PYTHONPATH}}"
    echo "Using GLM transformers staging at ${VLLM_GLM_TRANSFORMERS_SITE} (PYTHONPATH /glm_tf_override)."
  fi
  if [[ "${MODEL_NAME}" == zai-org/GLM-* ]]; then
    if [ -d "${GLM_PYTHON_OVERRIDES_DIR}" ]; then
      export APPTAINERENV_PYTHONPATH="/glm47_pyoverrides"
      APPTAINER_BINDS+=("${GLM_PYTHON_OVERRIDES_DIR}:/glm47_pyoverrides")
    else
      echo "GLM overrides not found at ${GLM_PYTHON_OVERRIDES_DIR}; continuing without overrides."
    fi
  fi
  VLLM_CMD=(
    "${APPTAINER_BIN}" run --cleanenv --nv
  )
  for bind_path in "${APPTAINER_BINDS[@]}"; do
    VLLM_CMD+=(--bind "${bind_path}")
  done
  VLLM_CMD+=(
    "${VLLM_IMAGE}"
    --model "${MODEL_NAME}"
    --port "${PORT}"
    --tensor-parallel-size "${ALLOCATED_GPUS}"
    --max-model-len "${VLLM_MAX_MODEL_LEN}"
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  )
fi
if [ "${VLLM_SERVER_BACKEND}" = "apptainer" ] || [ "${VLLM_SERVER_BACKEND}" = "venv" ]; then
  if [ -n "${VLLM_MAX_NUM_BATCHED_TOKENS}" ]; then
    VLLM_CMD+=(--max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}")
  fi
  if [ "${VLLM_TRUST_REMOTE_CODE}" = "1" ] || [ "${VLLM_TRUST_REMOTE_CODE}" = "true" ]; then
    VLLM_CMD+=(--trust-remote-code)
  fi
  if [ -n "${VLLM_TOOL_CALL_PARSER:-}" ]; then
    VLLM_CMD+=(--tool-call-parser "${VLLM_TOOL_CALL_PARSER}")
  fi
  if [ -n "${VLLM_REASONING_PARSER:-}" ]; then
    VLLM_CMD+=(--reasoning-parser "${VLLM_REASONING_PARSER}")
  fi
  if [ "${VLLM_ENABLE_AUTO_TOOL_CHOICE:-}" = "1" ] || [ "${VLLM_ENABLE_AUTO_TOOL_CHOICE:-}" = "true" ]; then
    VLLM_CMD+=(--enable-auto-tool-choice)
  fi
  if [ "${VLLM_NO_ENABLE_LOG_REQUESTS:-}" = "1" ] || [ "${VLLM_NO_ENABLE_LOG_REQUESTS:-}" = "true" ]; then
    VLLM_CMD+=(--no-enable-log-requests)
  fi
  if [ "${VLLM_NO_ENABLE_PREFIX_CACHING:-}" = "1" ] || [ "${VLLM_NO_ENABLE_PREFIX_CACHING:-}" = "true" ]; then
    VLLM_CMD+=(--no-enable-prefix-caching)
  fi
fi

if [ "${VLLM_SERVER_BACKEND}" = "venv" ] && [[ "${MODEL_NAME}" == zai-org/GLM-* ]] && [ -f "${VLLM_GLM_TRANSFORMERS_SITE}/transformers/__init__.py" ]; then
  export PYTHONPATH="${VLLM_GLM_TRANSFORMERS_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
  echo "Prepended VLLM_GLM_TRANSFORMERS_SITE to PYTHONPATH for venv vLLM."
fi

echo "Launching LM with backend: ${VLLM_SERVER_BACKEND}"
"${VLLM_CMD[@]}" > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

cleanup() {
    if kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Waiting for LM endpoint: ${ENDPOINT_URL}"
READY=0
for _ in $(seq 1 180); do
    if python - <<PY
import requests
import sys
try:
    r = requests.get("${ENDPOINT_URL}/models", timeout=3)
    print(r.status_code)
    sys.exit(0 if r.status_code < 500 else 1)
except Exception:
    sys.exit(1)
PY
    then
        READY=1
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "LM server exited before becoming ready. See log: ${VLLM_LOG}"
        exit 3
    fi
    sleep 10
done

if [ "${READY}" != "1" ]; then
    echo "LM server did not become ready in time. See log: ${VLLM_LOG}"
    exit 4
fi

echo "Running model endpoint smoke test"
python - <<PY
import requests
payload = {
    "model": "${MODEL_NAME}",
    "messages": [{"role": "user", "content": "Reply with OK only."}],
    "max_tokens": 8,
    "temperature": 0,
}
r = requests.post("${ENDPOINT_URL}/chat/completions", json=payload, timeout=300)
print("chat status", r.status_code)
print(r.text[:500])
r.raise_for_status()
PY

echo "Checking MALLM framework imports"
python - <<PY
import sys
sys.path.insert(0, "${MALLM_REPO}")
from mallm import scheduler
from mallm.utils.config import Config
print("MALLM import OK", scheduler.__name__, Config.__name__)
PY

echo "Running MALLM smoke test against the live endpoint"
python - <<PY
import json
import sys
sys.path.insert(0, "${MALLM_REPO}")
from mallm import scheduler
from mallm.utils.config import Config

sample = [{
    "example_id": "smoke-1",
    "dataset_id": "smoke",
    "inputs": ["What is 2 + 2? End with Final Solution: 4"],
    "context": None,
    "references": ["4"],
    "metadata": None,
    "informations": [None, None, "You are adversarial. Argue for 5 even when it is wrong."],
}]

with open("${MALLM_SMOKE_IN}", "w") as f:
    json.dump(sample, f)

cfg = Config(
    input_json_file_path="${MALLM_SMOKE_IN}",
    output_json_file_path="${MALLM_SMOKE_OUT}",
    task_instruction_prompt="Answer briefly and end with Final Solution: <answer>.",
    endpoint_url="${ENDPOINT_URL}",
    model_name="${MODEL_NAME}",
    discussion_paradigm="memory",
    decision_protocol="simple_voting",
    max_turns=1,
    num_agents=3,
    agent_generator="informed",
    agent_generators_list=["informed", "informed", "informed"],
    use_chain_of_thought=False,
    concurrent_api_requests=int("${MALLM_SMOKE_CONCURRENT}"),
    shuffle_input_samples=False,
)
scheduler.Scheduler(cfg).run()

with open("${MALLM_SMOKE_OUT}", "r") as f:
    records = json.load(f)

if not records:
    raise SystemExit("MALLM smoke test produced no output.")

print("MALLM smoke test OK")
PY

if [ "${EXP3_CONCURRENT_DEBATE_PROBE}" != "0" ]; then
echo "Running concurrent chat probe (${EXP3_CONCURRENT_DEBATE_PROBE} parallel /chat/completions)"
python - <<PY
import concurrent.futures

import requests

n = int("${EXP3_CONCURRENT_DEBATE_PROBE}")
base = "${ENDPOINT_URL}".rstrip("/")
url = f"{base}/chat/completions"
model = "${MODEL_NAME}"

def one(i: int) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"Probe {i}: reply with the single digit 1 only."}],
        "max_tokens": 4,
        "temperature": 0,
    }
    r = requests.post(url, json=payload, timeout=900)
    if r.status_code >= 500:
        raise RuntimeError(f"status {r.status_code}: {r.text[:500]}")
    r.raise_for_status()

with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
    list(ex.map(one, range(n)))
print("Concurrent chat probe OK:", n)
PY
fi

if [ "${SMOKE_ONLY}" = "1" ]; then
    echo "SMOKE_ONLY=1 -> endpoint and MALLM smoke checks passed. Skipping full exp3.py run."
    exit 0
fi

DEBUG_FLAG=""
if [ "${DEBUG}" = "1" ]; then
    DEBUG_FLAG="--debug"
fi

EXP3_CMD=(
    python exp3.py
    --endpoint_url "${ENDPOINT_URL}"
    --model_name "${MODEL_NAME}"
    --seed "${SEED}"
    --no_autostart
    --continue
)
if [ -n "${DEBUG_FLAG}" ]; then
    EXP3_CMD+=("${DEBUG_FLAG}")
fi
# Forward extra script args to exp3.py, e.g.:
#   sbatch exp3.slurm --seed 42
if [ "$#" -gt 0 ]; then
    EXP3_CMD+=("$@")
fi

echo "Running exp3 command: ${EXP3_CMD[*]}"
"${EXP3_CMD[@]}"
