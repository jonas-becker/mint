"""
Shared model-name normalization and vLLM serving defaults for exp1a/exp2/exp3.

Aligned with upstream recipes:
  - GLM: https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM.html
  - GPT-OSS: https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html
"""

from __future__ import annotations

import os
from typing import Any, List, Optional


def normalize_model_name(name: str) -> str:
    """Map CLI/Slurm aliases to Hugging Face hub ids."""
    if not isinstance(name, str):
        return ""
    raw = name.strip()
    key = raw.lower().replace("_", "-")
    aliases = {
        "llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
        "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "openai/gpt-oss-120b": "openai/gpt-oss-120b",
        "gpt-oss-20b": "openai/gpt-oss-20b",
        "openai/gpt-oss-20b": "openai/gpt-oss-20b",
        "glm-4.7-flash": "zai-org/GLM-4.7-Flash",
        "zai-org/glm-4.7-flash": "zai-org/GLM-4.7-Flash",
    }
    return aliases.get(key, raw)


def is_glm_model(model_name: str) -> bool:
    return (model_name or "").startswith("zai-org/GLM-")


def is_gpt_oss_model(model_name: str) -> bool:
    m = (model_name or "").lower()
    return m.startswith("openai/gpt-oss") or "gpt-oss" in m


def is_llama_70b_model(model_name: str) -> bool:
    m = (model_name or "").lower()
    return "llama-3.3-70b" in m or "llama-3.1-70b" in m or "llama3.3-70b" in m


def min_gpus_required(model_name: str) -> int:
    """Minimum tensor-parallel GPUs for BF16/quantized weights on 80GB A100s."""
    if is_llama_70b_model(model_name):
        return 2
    return 1


def vllm_max_model_len_default(model_name: str) -> int:
    if is_glm_model(model_name):
        return 65536
    if is_gpt_oss_model(model_name):
        # Full context is 131072; use a practical default for KV cache on 80GB nodes.
        return 32768
    if is_llama_70b_model(model_name):
        return 8192
    return 8192


def vllm_gpu_memory_utilization_default(model_name: str) -> float:
    if is_glm_model(model_name) or is_llama_70b_model(model_name) or is_gpt_oss_model(model_name):
        return 0.90
    return 0.98


def vllm_trust_remote_code_default(model_name: str) -> bool:
    if is_glm_model(model_name) or (model_name or "").startswith("meta-llama/"):
        return True
    return False


def vllm_tool_call_parser_default(model_name: str) -> str:
    if is_gpt_oss_model(model_name):
        return "openai"
    if not is_glm_model(model_name):
        return ""
    m = model_name.lower()
    if "glm-4.5" in m or "glm4.5" in m:
        return "glm45"
    if "glm-4.6" in m or "glm4.6" in m or "glm-4.7" in m or "glm4.7" in m or "glm-5" in m or "glm5" in m:
        return "glm47"
    return ""


def vllm_reasoning_parser_default(model_name: str) -> str:
    # Do NOT default reasoning-parser for MALLM: it routes text into delta.reasoning /
    # reasoning_content and leaves delta.content empty. MALLM concatenates content first
    # but exp1a also reads reasoning_content as fallback.
    return ""


def vllm_enable_auto_tool_choice_default(model_name: str) -> bool:
    return is_glm_model(model_name) or is_gpt_oss_model(model_name)


def vllm_no_enable_log_requests_default(model_name: str) -> bool:
    return is_glm_model(model_name) or is_gpt_oss_model(model_name)


def vllm_no_enable_prefix_caching_default(model_name: str) -> bool:
    return is_glm_model(model_name)


def sglang_tool_call_parser_default(model_name: str) -> str:
    if is_gpt_oss_model(model_name):
        return ""
    if not is_glm_model(model_name):
        return ""
    m = model_name.lower()
    if "glm-4.5" in m or "glm4.5" in m:
        return "glm45"
    if "glm-4.6" in m or "glm4.6" in m or "glm-4.7" in m or "glm4.7" in m:
        return "glm47"
    if "glm-5" in m or "glm5" in m:
        return "glm47"
    return ""


def vllm_extra_args_for_model(model_name: str) -> List[str]:
    """
    Optional vLLM flags per model family. Env overrides:
      VLLM_TOOL_CALL_PARSER, VLLM_REASONING_PARSER,
      VLLM_ENABLE_AUTO_TOOL_CHOICE, VLLM_NO_ENABLE_LOG_REQUESTS,
      VLLM_NO_ENABLE_PREFIX_CACHING
    """
    extra: List[str] = []
    tcp = (os.environ.get("VLLM_TOOL_CALL_PARSER") or "").strip()
    if not tcp:
        tcp = vllm_tool_call_parser_default(model_name)
    if tcp:
        extra += ["--tool-call-parser", tcp]

    rp = (os.environ.get("VLLM_REASONING_PARSER") or "").strip()
    if not rp:
        rp = vllm_reasoning_parser_default(model_name)
    if rp:
        extra += ["--reasoning-parser", rp]

    eatc = (os.environ.get("VLLM_ENABLE_AUTO_TOOL_CHOICE") or "").strip().lower()
    if not eatc and vllm_enable_auto_tool_choice_default(model_name):
        eatc = "1"
    if eatc in ("1", "true", "yes"):
        extra.append("--enable-auto-tool-choice")

    nlr = (os.environ.get("VLLM_NO_ENABLE_LOG_REQUESTS") or "").strip().lower()
    if not nlr and vllm_no_enable_log_requests_default(model_name):
        nlr = "1"
    if nlr in ("1", "true", "yes"):
        extra.append("--no-enable-log-requests")

    npc = (os.environ.get("VLLM_NO_ENABLE_PREFIX_CACHING") or "").strip().lower()
    if not npc and vllm_no_enable_prefix_caching_default(model_name):
        npc = "1"
    if npc in ("1", "true", "yes"):
        extra.append("--no-enable-prefix-caching")

    return extra


def extract_openai_message_text(message: Any) -> str:
    """Best-effort assistant text from a chat completion message (gpt-oss aware)."""
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    for attr in ("reasoning_content", "reasoning"):
        alt = getattr(message, attr, None)
        if isinstance(alt, str) and alt.strip():
            return alt.strip()
    if content is not None:
        return str(content).strip()
    return ""


def mallm_max_tokens_default(_model_name: str) -> int:
    """Model-agnostic MALLM generation cap unless overridden by env."""
    raw = (os.environ.get("MALLM_MAX_TOKENS") or "").strip()
    if raw.isdigit():
        return max(1, int(raw))
    return 512


def gpt_oss_tiktoken_dir() -> str:
    """Host directory with o200k_base.tiktoken and cl100k_base.tiktoken for offline gpt-oss."""
    candidates = [
        (os.environ.get("GPT_OSS_TIKTOKEN_DIR") or "").strip(),
        "${TIKTOKEN_ENCODINGS_DIR:-${SHARED_HF_CACHE_VOLUME}/tiktoken_encodings}",
        os.path.expanduser("~/.cache/tiktoken_encodings"),
    ]
    required = ("o200k_base.tiktoken", "cl100k_base.tiktoken")
    for directory in candidates:
        if not directory:
            continue
        if all(os.path.isfile(os.path.join(directory, name)) for name in required):
            return directory
    return ""


def apply_gpt_oss_tiktoken_env(model_name: str) -> None:
    """Set TIKTOKEN_* env vars required for offline vLLM gpt-oss serving."""
    if not is_gpt_oss_model(model_name):
        return
    directory = gpt_oss_tiktoken_dir()
    if not directory:
        raise RuntimeError(
            "gpt-oss requires offline tiktoken vocab files. "
            "Run: bash scripts/download_gpt_oss_tiktoken.sh"
        )
    os.environ["TIKTOKEN_ENCODINGS_BASE"] = directory
    os.environ["TIKTOKEN_RS_CACHE_DIR"] = directory


def mallm_concurrency_default(_model_name: str) -> int:
    raw = (os.environ.get("MALLM_CONCURRENT_API_REQUESTS") or "").strip()
    if raw.isdigit():
        return max(1, int(raw))
    return 64
