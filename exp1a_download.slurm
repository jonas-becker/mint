#!/bin/bash -l
#SBATCH --job-name=exp1a_dl70b
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --nodes=1
# Download meta-llama/Llama-3.3-70B-Instruct to Ceph (login/medium nodes have HF network).
# Then run: sbatch --gpus=2 --export=ALL,HF_TOKEN exp1a.slurm
#
#   sbatch --export=ALL,HF_TOKEN exp1a_download.slurm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
cd "${REPO_ROOT}"

# Same default as exp2_ablation.slurm (override with export HF_TOKEN=... if needed).
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before submitting}"
export HF_TOKEN

PROJECT_HOME="${PROJECT_HOME:-${HOME}}"
VENV_PATH="${VENV_PATH:-${PROJECT_HOME}/.venvs/misinformed_agents_exp2}"
HF_CACHE_VOLUME="${HF_CACHE_VOLUME:-${LLAMA_70B_HF_CACHE:-${HF_HOME}/cache}}"

if command -v module >/dev/null 2>&1; then
    module load env/staging/2024.1 >/dev/null 2>&1 || true
    module load Python/3.11.10-GCCcore-13.3.0 >/dev/null 2>&1 || true
fi

if [ ! -d "${VENV_PATH}" ]; then
    echo "Expected venv at ${VENV_PATH}; create it or set VENV_PATH."
    exit 2
fi
source "${VENV_PATH}/bin/activate"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: Set HF_TOKEN (export HF_TOKEN=hf_... or sbatch --export=ALL,HF_TOKEN=...)"
    exit 2
fi

if ! curl -sf --connect-timeout 10 https://huggingface.co >/dev/null; then
    echo "ERROR: Cannot reach huggingface.co from this node. Run on a login/medium node with network."
    exit 2
fi

export HF_CACHE_VOLUME
bash "${REPO_ROOT}/scripts/download_llama_33_70b.sh"
echo "Download job finished. Submit experiment with:"
echo "  sbatch --gpus=2 --export=ALL,HF_TOKEN exp1a.slurm"
