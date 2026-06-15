#!/usr/bin/env bash
# Download OpenAI tiktoken vocab files required for offline vLLM gpt-oss serving.
# See: https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${GPT_OSS_TIKTOKEN_DIR:-${HOME}/.cache/tiktoken_encodings}"
SHARED_DIR="${SHARED_HF_CACHE_VOLUME:-${SHARED_HF_CACHE_VOLUME:-/shared/hf_cache}}/tiktoken_encodings"

mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

for name in o200k_base.tiktoken cl100k_base.tiktoken; do
    if [ -f "${name}" ]; then
        echo "Already present: ${TARGET_DIR}/${name}"
        continue
    fi
    echo "Downloading ${name}..."
    wget -O "${name}" "https://openaipublic.blob.core.windows.net/encodings/${name}"
done

echo "Tiktoken encodings ready at ${TARGET_DIR}"
if [ -w "$(dirname "${SHARED_DIR}")" ] 2>/dev/null; then
    mkdir -p "${SHARED_DIR}"
    cp -f o200k_base.tiktoken cl100k_base.tiktoken "${SHARED_DIR}/"
    echo "Copied to shared store: ${SHARED_DIR}"
fi
