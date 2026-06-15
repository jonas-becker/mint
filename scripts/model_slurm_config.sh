#!/usr/bin/env bash
# Shared model normalization and vLLM defaults for exp1a/exp2/exp3 Slurm scripts.
# Source after REPO_ROOT / HF_CACHE_VOLUME / SHARED_HF_CACHE_VOLUME are set.

is_hub_model_id() {
    [[ "$1" == */* ]] && [ ! -d "$1" ]
}

normalize_model_name() {
    local raw="${1:-}"
    local key="${raw,,}"
    case "${key}" in
        llama-3.3-70b-instruct|meta-llama/llama-3.3-70b-instruct)
            echo "meta-llama/Llama-3.3-70B-Instruct"
            ;;
        llama-3.1-8b-instruct|meta-llama/llama-3.1-8b-instruct)
            echo "meta-llama/Llama-3.1-8B-Instruct"
            ;;
        llama-3.2-3b-instruct|meta-llama/llama-3.2-3b-instruct)
            echo "meta-llama/Llama-3.2-3B-Instruct"
            ;;
        gpt-oss-120b|openai/gpt-oss-120b)
            echo "openai/gpt-oss-120b"
            ;;
        gpt-oss-20b|openai/gpt-oss-20b)
            echo "openai/gpt-oss-20b"
            ;;
        glm-4.7-flash|zai-org/glm-4.7-flash)
            echo "zai-org/GLM-4.7-Flash"
            ;;
        *)
            echo "${raw}"
            ;;
    esac
}

is_glm_model() {
    [[ "${1:-}" == zai-org/GLM-* ]]
}

is_gpt_oss_model() {
    local m="${1,,}"
    [[ "${m}" == openai/gpt-oss* ]] || [[ "${m}" == *gpt-oss* ]]
}

is_llama_70b_model() {
    local m="${1,,}"
    [[ "${m}" == *"llama-3.3-70b"* ]] || [[ "${m}" == *"llama-3.1-70b"* ]] || [[ "${m}" == *"llama3.3-70b"* ]]
}

min_gpus_required_for_model() {
    if is_llama_70b_model "$1"; then
        echo 2
    else
        echo 1
    fi
}

is_model_cached_in() {
    local model_id="$1"
    local cache_root="$2"
    local model_dir="${cache_root}/hub/models--${model_id//\//--}/snapshots"
    [ -d "${model_dir}" ] || return 1
    ls -A "${model_dir}" >/dev/null 2>&1
}

is_model_cached_anywhere() {
    is_model_cached_in "$1" "${HF_CACHE_VOLUME}" || is_model_cached_in "$1" "${SHARED_HF_CACHE_VOLUME}"
}

apply_model_hf_cache_path() {
    # Llama-70B: optional personal Ceph override. All other hub models: symlink from shared when missing locally.
    local model_id="$1"
    if is_llama_70b_model "${model_id}" && [ "${HF_CACHE_VOLUME}" = "${_HF_CACHE_HOME_DEFAULT}" ]; then
        HF_CACHE_VOLUME="${LLAMA_70B_HF_CACHE:-${LLAMA_70B_HF_CACHE:-${HF_HOME}/cache}}"
        echo "[INFO] Llama-70B: HF_CACHE_VOLUME=${HF_CACHE_VOLUME}"
    fi
    if is_hub_model_id "${model_id}"; then
        if ! is_model_cached_in "${model_id}" "${HF_CACHE_VOLUME}" \
            && is_model_cached_in "${model_id}" "${SHARED_HF_CACHE_VOLUME}"; then
            SHARED_MODEL_CACHE_SOURCE="${SHARED_HF_CACHE_VOLUME}/hub/models--${model_id//\//--}"
            LOCAL_MODEL_CACHE_TARGET="${HF_CACHE_VOLUME}/hub/models--${model_id//\//--}"
            echo "Will link model cache from shared store:"
            echo "  source: ${SHARED_MODEL_CACHE_SOURCE}"
            echo "  target: ${LOCAL_MODEL_CACHE_TARGET}"
        fi
    fi
}

apply_model_vllm_max_model_len_default() {
    local model_id="$1"
    if [ -n "${VLLM_MAX_MODEL_LEN:-}" ]; then
        return 0
    fi
    if is_glm_model "${model_id}"; then
        VLLM_MAX_MODEL_LEN="65536"
    elif is_gpt_oss_model "${model_id}"; then
        VLLM_MAX_MODEL_LEN="32768"
    elif is_llama_70b_model "${model_id}"; then
        VLLM_MAX_MODEL_LEN="8192"
    else
        VLLM_MAX_MODEL_LEN="8192"
    fi
}

apply_model_vllm_trust_remote_code_default() {
    local model_id="$1"
    if [ -n "${VLLM_TRUST_REMOTE_CODE}" ]; then
        return 0
    fi
    if is_glm_model "${model_id}" || [[ "${model_id}" == meta-llama/* ]]; then
        VLLM_TRUST_REMOTE_CODE="1"
    else
        VLLM_TRUST_REMOTE_CODE="0"
    fi
}

apply_model_vllm_gpu_mem_default() {
    local model_id="$1"
    if [ -n "${VLLM_GPU_MEMORY_UTILIZATION}" ]; then
        return 0
    fi
    if is_glm_model "${model_id}" || is_llama_70b_model "${model_id}" || is_gpt_oss_model "${model_id}"; then
        VLLM_GPU_MEMORY_UTILIZATION="0.90"
    else
        VLLM_GPU_MEMORY_UTILIZATION="0.98"
    fi
}

apply_model_vllm_glm_gptoss_defaults() {
    local model_id="$1"
    if is_glm_model "${model_id}"; then
        export APPTAINERENV_SAFETENSORS_FAST_GPU="${APPTAINERENV_SAFETENSORS_FAST_GPU:-1}"
        if [ -z "${VLLM_ENABLE_AUTO_TOOL_CHOICE}" ]; then
            VLLM_ENABLE_AUTO_TOOL_CHOICE="1"
        fi
        if [ -z "${VLLM_NO_ENABLE_LOG_REQUESTS}" ]; then
            VLLM_NO_ENABLE_LOG_REQUESTS="1"
        fi
        if [ -z "${VLLM_NO_ENABLE_PREFIX_CACHING}" ]; then
            VLLM_NO_ENABLE_PREFIX_CACHING="1"
        fi
        if [ -z "${VLLM_TOOL_CALL_PARSER}" ]; then
            case "${model_id}" in
                zai-org/GLM-4.5*)
                    VLLM_TOOL_CALL_PARSER="glm45"
                    ;;
                zai-org/GLM-4.6*|zai-org/GLM-4.7*|zai-org/GLM-5*)
                    VLLM_TOOL_CALL_PARSER="glm47"
                    ;;
            esac
        fi
    elif is_gpt_oss_model "${model_id}"; then
        if [ -z "${VLLM_ENABLE_AUTO_TOOL_CHOICE}" ]; then
            VLLM_ENABLE_AUTO_TOOL_CHOICE="1"
        fi
        if [ -z "${VLLM_NO_ENABLE_LOG_REQUESTS}" ]; then
            VLLM_NO_ENABLE_LOG_REQUESTS="1"
        fi
        if [ -z "${VLLM_TOOL_CALL_PARSER}" ]; then
            VLLM_TOOL_CALL_PARSER="openai"
        fi
        # Do NOT set VLLM_REASONING_PARSER: MALLM/exp1a read content (+ reasoning_content fallback).
        # gpt-oss is slower per request; use a modest smoke probe unless explicitly overridden.
        if [ "${EXP1A_CONCURRENT_PROBE:-64}" = "64" ]; then
            EXP1A_CONCURRENT_PROBE="16"
        fi
        if [ "${EXP1_CONCURRENT_PROBE:-64}" = "64" ]; then
            EXP1_CONCURRENT_PROBE="16"
        fi
        if [ "${EXP2_CONCURRENT_DEBATE_PROBE:-64}" = "64" ]; then
            EXP2_CONCURRENT_DEBATE_PROBE="16"
        fi
        if [ "${EXP3_CONCURRENT_DEBATE_PROBE:-64}" = "64" ]; then
            EXP3_CONCURRENT_DEBATE_PROBE="16"
        fi
    fi
}

apply_model_sglang_tool_parser_default() {
    local model_id="$1"
    if [ -n "${SGLANG_TOOL_CALL_PARSER:-}" ]; then
        return 0
    fi
    if is_glm_model "${model_id}"; then
        case "${model_id}" in
            zai-org/GLM-4.5*)
                SGLANG_TOOL_CALL_PARSER="glm45"
                ;;
            zai-org/GLM-4.6*|zai-org/GLM-4.7*|zai-org/GLM-5*)
                SGLANG_TOOL_CALL_PARSER="glm47"
                ;;
        esac
    fi
}

ensure_model_gpu_allocation() {
    local model_id="$1"
    local min_gpus
    min_gpus="$(min_gpus_required_for_model "${model_id}")"
    if [ "${ALLOCATED_GPUS}" -lt "${min_gpus}" ]; then
        echo "ERROR: ${model_id} requires at least ${min_gpus} GPU(s); job has ${ALLOCATED_GPUS}."
        if is_llama_70b_model "${model_id}"; then
            echo "Submit with: sbatch --gpus=2 exp1a.slurm"
        elif is_gpt_oss_model "${model_id}"; then
            echo "gpt-oss-120b fits on 1×80GB A100; submit with: sbatch --gpus=1 ..."
        fi
        exit 2
    fi
    if is_gpt_oss_model "${model_id}" && [ "${ALLOCATED_GPUS}" -gt 1 ]; then
        echo "[INFO] gpt-oss: using tensor-parallel-size=${ALLOCATED_GPUS} (1 GPU is sufficient on 80GB A100)."
    fi
}

gpt_oss_tiktoken_dir() {
    local candidates=(
        "${GPT_OSS_TIKTOKEN_DIR:-}"
        "${SHARED_HF_CACHE_VOLUME}/tiktoken_encodings"
        "${PROJECT_HOME:-${HOME}}/.cache/tiktoken_encodings"
    )
    local dir
    for dir in "${candidates[@]}"; do
        [ -n "${dir}" ] || continue
        if [ -f "${dir}/o200k_base.tiktoken" ] && [ -f "${dir}/cl100k_base.tiktoken" ]; then
            echo "${dir}"
            return 0
        fi
    done
    return 1
}

apply_gpt_oss_tiktoken_env() {
    # vLLM gpt-oss needs OpenAI harmony tiktoken vocabs offline (see vLLM GPT-OSS recipe).
    local model_id="$1"
    is_gpt_oss_model "${model_id}" || return 0
    local host_dir
    if ! host_dir="$(gpt_oss_tiktoken_dir)"; then
        echo "ERROR: gpt-oss requires offline tiktoken vocab files (o200k_base.tiktoken, cl100k_base.tiktoken)."
        echo "On a login node with network:"
        echo "  bash ${REPO_ROOT:-.}/scripts/download_gpt_oss_tiktoken.sh"
        echo "Or set GPT_OSS_TIKTOKEN_DIR to a directory containing both files."
        exit 2
    fi
    GPT_OSS_TIKTOKEN_BIND="${host_dir}:/tiktoken_encodings:ro"
    export TIKTOKEN_ENCODINGS_BASE="/tiktoken_encodings"
    export TIKTOKEN_RS_CACHE_DIR="/tiktoken_encodings"
    export APPTAINERENV_TIKTOKEN_ENCODINGS_BASE="/tiktoken_encodings"
    export APPTAINERENV_TIKTOKEN_RS_CACHE_DIR="/tiktoken_encodings"
    echo "[INFO] gpt-oss tiktoken encodings: ${host_dir} -> /tiktoken_encodings"
}

ensure_hub_model_cached_or_exit() {
    local model_id="$1"
    is_hub_model_id "${model_id}" || return 0
    if is_model_cached_anywhere "${model_id}"; then
        return 0
    fi
    echo "ERROR: ${model_id} is not in the local or shared Hugging Face cache."
    echo "Compute nodes cannot reach huggingface.co."
    echo "Shared cache: ${SHARED_HF_CACHE_VOLUME}/hub/models--${model_id//\//--}"
    echo "Local cache:  ${HF_CACHE_VOLUME}/hub/models--${model_id//\//--}/snapshots"
    if is_gpt_oss_model "${model_id}"; then
        echo "gpt-oss-120b is available on shared store; ensure HF_CACHE_VOLUME symlinks are created (see apply_model_hf_cache_path)."
    fi
    exit 2
}
