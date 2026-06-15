#!/usr/bin/env python3

"""
exp3.py - Multi-Agent Debate Experiment (MALLM): varying (mis)informed agent composition.

This script is intentionally similar to exp2.py, but instead of sweeping misinformation
strategies across multiple datasets, it:
  - runs ONLY winogrande
  - runs 6 composition settings with 5 total agents:
      0 misinformed, 5 informed
      1 misinformed, 4 informed
      2 misinformed, 3 informed
      3 misinformed, 2 informed
      4 misinformed, 1 informed
      5 misinformed, 0 informed

Implementation detail:
MALLM's "informed" persona generator reads per-agent information from
InputExample.informations[idx]. We treat an agent as "misinformed" if its info slot
contains a misinformation string, otherwise "informed" (None).
"""

import argparse
import atexit
import glob
import hashlib
import json
import os
import re
import shutil
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid
from typing import List, Optional
from datetime import datetime, timezone

from model_server_config import (
    apply_gpt_oss_tiktoken_env,
    is_glm_model,
    normalize_model_name,
    sglang_tool_call_parser_default,
    vllm_extra_args_for_model,
    vllm_gpu_memory_utilization_default,
    vllm_max_model_len_default,
    vllm_trust_remote_code_default,
)

if sys.version_info < (3, 7):
    raise SystemExit(
        "exp3.py requires Python >= 3.7 (MALLM depends on the stdlib 'dataclasses'). "
        "On this system, try: python3.11 ${REPO_ROOT}/exp3.py ..."
    )


#
# Lightweight integration with MALLM (mirrors exp2.py)
#
def _ensure_mallm_on_path() -> None:
    """
    Ensure the local MALLM repository is importable without installation.
    Expected layout:
      github/misinformed_agents/exp3.py
      github/mallm/...
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    mallm_repo_root = os.path.abspath(os.path.join(this_dir, "..", "mallm"))
    if mallm_repo_root not in sys.path:
        sys.path.insert(0, mallm_repo_root)


def _safe_model_name(model_name: str) -> str:
    safe = (model_name or "").strip().replace("/", "_")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", safe)
    return safe or "unknown_model"


def _install_runtime_stubs() -> None:
    """
    Install lightweight stubs for optional heavy dependencies so that we can run
    a mock discussion without installing large packages.
    (Copied from exp2.py with minimal changes.)
    """
    import types
    import logging as _logging

    # Stub: langchain_core
    if "langchain_core" not in sys.modules:
        langchain_core = types.ModuleType("langchain_core")
        sys.modules["langchain_core"] = langchain_core
        callbacks = types.ModuleType("langchain_core.callbacks")

        class Callbacks:  # noqa: N801
            pass

        class CallbackManagerForLLMRun:  # noqa: N801
            pass

        callbacks.Callbacks = Callbacks
        callbacks.CallbackManagerForLLMRun = CallbackManagerForLLMRun
        sys.modules["langchain_core.callbacks"] = callbacks

        callbacks_manager = types.ModuleType("langchain_core.callbacks.manager")
        callbacks_manager.CallbackManagerForLLMRun = CallbackManagerForLLMRun
        sys.modules["langchain_core.callbacks.manager"] = callbacks_manager

        language_models = types.ModuleType("langchain_core.language_models")

        class LanguageModelInput:  # noqa: N801
            pass

        language_models.LanguageModelInput = LanguageModelInput
        sys.modules["langchain_core.language_models"] = language_models

        llms = types.ModuleType("langchain_core.language_models.llms")

        class LLM:  # noqa: N801
            def __init__(self, *args, **kwargs) -> None:
                for k, v in kwargs.items():
                    setattr(self, k, v)

            def invoke(self, prompt, **kwargs):  # type: ignore[override]
                if hasattr(self, "_call"):
                    return self._call(prompt, **kwargs)  # type: ignore[attr-defined]
                return ""

        llms.LLM = LLM
        sys.modules["langchain_core.language_models.llms"] = llms

        outputs = types.ModuleType("langchain_core.outputs")

        class LLMResult:  # noqa: N801
            pass

        outputs.LLMResult = LLMResult
        sys.modules["langchain_core.outputs"] = outputs

        prompt_values = types.ModuleType("langchain_core.prompt_values")

        class PromptValue:  # noqa: N801
            pass

        prompt_values.PromptValue = PromptValue
        sys.modules["langchain_core.prompt_values"] = prompt_values

    # Stub: langchain
    if "langchain" not in sys.modules:
        sys.modules["langchain"] = types.ModuleType("langchain")

    # Stub: httpx minimal client (only if package is unavailable)
    if "httpx" not in sys.modules:
        try:
            import httpx  # noqa: F401
        except Exception:
            httpx = types.ModuleType("httpx")

            class HTTPError(Exception):  # noqa: N801
                pass

            class Client:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

            httpx.HTTPError = HTTPError
            httpx.Client = Client
            sys.modules["httpx"] = httpx

    # Stub: openai.OpenAI (imported by mallm even for mock://)
    if "openai" not in sys.modules:
        openai = types.ModuleType("openai")

        class APIError(Exception):  # noqa: N801
            pass

        class APIConnectionError(Exception):  # noqa: N801
            pass

        class RateLimitError(Exception):  # noqa: N801
            pass

        class OpenAI:  # noqa: N801
            def __init__(self, *args, **kwargs):
                pass

        openai.APIError = APIError
        openai.APIConnectionError = APIConnectionError
        openai.RateLimitError = RateLimitError
        openai.OpenAI = OpenAI
        sys.modules["openai"] = openai

    # Stub: rich
    if "rich" not in sys.modules:
        rich = types.ModuleType("rich")

        def _rprint(*args, **kwargs):
            print(*args, **kwargs)

        rich.print = _rprint  # type: ignore[attr-defined]
        sys.modules["rich"] = rich

        rich_logging = types.ModuleType("rich.logging")

        class RichHandler(_logging.Handler):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def emit(self, record):  # type: ignore[override]
                pass

        rich_logging.RichHandler = RichHandler
        sys.modules["rich.logging"] = rich_logging

        rich_progress = types.ModuleType("rich.progress")

        class Console:
            def __init__(self, record: bool = False):
                self.width = 100

            def save_html(self, path: str, clear: bool = False):
                pass

            def print(self, *args, **kwargs):
                pass

        class Progress:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def add_task(self, *args, **kwargs):
                return 0

            def update(self, *args, **kwargs):
                pass

        TaskID = int
        rich_progress.Console = Console
        rich_progress.Progress = Progress
        rich_progress.TaskID = TaskID
        sys.modules["rich.progress"] = rich_progress

        rich_panel = types.ModuleType("rich.panel")

        class Panel:
            def __init__(self, *args, **kwargs):
                pass

        rich_panel.Panel = Panel
        sys.modules["rich.panel"] = rich_panel

        rich_text = types.ModuleType("rich.text")

        class Text:
            def __init__(self, *args, **kwargs):
                pass

            def highlight_regex(self, *args, **kwargs):
                pass

        rich_text.Text = Text
        sys.modules["rich.text"] = rich_text

    # Stub: contextplus.context
    if "contextplus" not in sys.modules:
        contextplus = types.ModuleType("contextplus")

        def context(input_data: str) -> str:
            return ""

        contextplus.context = context  # type: ignore[attr-defined]
        sys.modules["contextplus"] = contextplus

    # Stub: torch (MALLM type imports reference torch.Tensor)
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")

        class Tensor:  # noqa: N801
            pass

        class _Cuda:
            @staticmethod
            def device_count() -> int:
                return 0

        torch.Tensor = Tensor  # type: ignore[attr-defined]
        torch.cuda = _Cuda()  # type: ignore[attr-defined]
        sys.modules["torch"] = torch


def _read_json(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def _choose_misinformation(item: dict) -> str:
    """
    Prefer 'false_fact'. If unavailable, choose the first available strategy text.
    """
    if isinstance(item.get("false_fact"), str) and item["false_fact"].strip():
        return item["false_fact"]
    strat_map = item.get("misinformation_by_strategy") or {}
    if isinstance(strat_map, dict) and strat_map:
        for key in sorted(k for k in strat_map.keys() if isinstance(k, str)):
            val = strat_map.get(key)
            if isinstance(val, str) and val.strip():
                return val
    return (
        "You are adversarial. Argue confidently for an incorrect answer using plausible-sounding but false claims. "
        "Avoid agreeing unless the group adopts your incorrect answer."
    )


def _stable_shuffle_informations(
    informations: List[Optional[str]],
    example_id: str,
    misinformed_count: int,
    global_seed: int,
) -> List[Optional[str]]:
    """
    Deterministically shuffle per-example to avoid positional bias while keeping runs reproducible.
    """
    seed_material = f"{example_id}::{misinformed_count}::{global_seed}"
    seed_int = int(hashlib.md5(seed_material.encode("utf-8")).hexdigest()[:8], 16)
    rng = __import__("random").Random(seed_int)
    out = list(informations)
    rng.shuffle(out)
    return out


def _convert_winogrande_for_mix(
    src_path: str,
    dst_path: str,
    misinformed_count: int,
    num_agents: int,
    debug_mode: bool,
    seed: int,
) -> None:
    """
    Convert winogrande_misinformed.json to MALLM InputExample list, with
    informations sized for `num_agents` and `misinformed_count` misinformation slots.
    """
    data = _read_json(src_path)
    if debug_mode and isinstance(data, list):
        data = data[:5]

    out = []
    for it in data:
        sentence = it.get("sentence", "")
        options = it.get("options", [])
        answer = it.get("answer")
        try:
            idx = int(answer) - 1 if isinstance(answer, str) else int(answer)
        except Exception:
            idx = 0
        correct = ""
        if isinstance(options, list) and 0 <= idx < len(options):
            correct = str(options[idx])
        input_str = sentence
        if isinstance(options, list) and len(options) >= 2:
            input_str += "\nOptions:\nA) " + str(options[0]) + "\nB) " + str(options[1])

        misinfo = _choose_misinformation(it)
        informations: List[Optional[str]] = [misinfo] * misinformed_count + [None] * (num_agents - misinformed_count)
        informations = _stable_shuffle_informations(
            informations=informations,
            example_id=str(it.get("id", it.get("ID", "")) or str(uuid.uuid4())),
            misinformed_count=misinformed_count,
            global_seed=seed,
        )

        out.append(
            {
                "example_id": it.get("id", str(uuid.uuid4())),
                "dataset_id": "winogrande",
                "inputs": [input_str],
                "context": None,
                "references": [correct] if correct else [],
                "metadata": {
                    "misinformed_count": misinformed_count,
                    "informed_count": (num_agents - misinformed_count),
                    "num_agents": num_agents,
                },
                "informations": informations,
            }
        )

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w") as f:
        json.dump(out, f)


def run_with_mallm(
    mock: bool,
    endpoint_url: str,
    model_name: str,
    debug_mode: bool = False,
    num_agents: int = 5,
    seed: int = 0,
    continue_mode: bool = False,
) -> None:
    """
    Run a 5-agent debate using MALLM on winogrande with varying misinformed-agent counts.
    """
    _ensure_mallm_on_path()
    if mock:
        _install_runtime_stubs()

    from mallm import scheduler
    from mallm.utils.config import Config

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_root_dir = os.path.join(script_dir, "out")
    os.makedirs(out_root_dir, exist_ok=True)
    model_name_suffix = "mock_model" if mock else _safe_model_name(model_name)
    out_dir = os.path.join(out_root_dir, model_name_suffix)
    os.makedirs(out_dir, exist_ok=True)
    mallm_io_dir = os.path.join(out_dir, "exp3_mallm_io")
    os.makedirs(mallm_io_dir, exist_ok=True)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(os.path.dirname(repo_root), "misinformed_agents", "data")
    src = os.path.join(data_dir, "winogrande_misinformed.json")
    if not os.path.exists(src):
        raise FileNotFoundError(f"Dataset not found: {src}")

    # Determine how many samples we expect per mix for resume logic
    try:
        raw_data = _read_json(src)
        expected_total = len(raw_data[:5]) if debug_mode else len(raw_data)
    except Exception:
        expected_total = 5 if debug_mode else None

    out_path = os.path.join(out_dir, "exp3_results_winogrande.json")

    def _load_existing() -> dict:
        if not os.path.exists(out_path):
            return {}
        try:
            with open(out_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _evaluate_and_aggregate(output_path: str) -> dict:
        try:
            with open(output_path, "r") as f:
                records = json.load(f)
        except Exception:
            records = []

        results = []
        correct_count = 0
        total = 0
        for rec in records:
            inputs = rec.get("input") or []
            input_str = inputs[0] if isinstance(inputs, list) and inputs else ""
            refs = rec.get("references") or []
            final_answer = rec.get("finalAnswer")
            predicted = str(final_answer) if final_answer is not None else ""

            is_correct = False
            if isinstance(refs, list) and refs:
                corr_text = str(refs[0]).lower()
                is_correct = corr_text in predicted.lower()

            total += 1
            if is_correct:
                correct_count += 1

            results.append(
                {
                    "sentence": input_str,
                    "response": predicted,
                    "predicted_answer": predicted,
                    "correct_answer": (refs[0] if isinstance(refs, list) and refs else ""),
                    "is_correct": is_correct,
                    "is_multiple_choice": True,
                    "options": None,
                    "misinformation_strategy": "false_fact",
                    "mallm_log": rec,
                }
            )

        accuracy = (correct_count / total) if total else 0.0
        return {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total,
            "results": results,
        }

    def _save_aggregate(mix_key: str, mix_results: dict) -> None:
        existing = _load_existing()
        existing[mix_key] = mix_results
        with open(out_path, "w") as f:
            json.dump(existing, f)

    # Six mixes requested by user
    mixes = list(range(0, num_agents + 1))

    existing_results = _load_existing() if continue_mode else {}

    for misinformed_count in mixes:
        informed_count = num_agents - misinformed_count
        mix_key = f"{misinformed_count}_misinformed_{informed_count}_informed"

        if continue_mode and mix_key in existing_results:
            prev = existing_results.get(mix_key, {}) if isinstance(existing_results, dict) else {}
            prev_total = prev.get("total_count")
            prev_cfg = prev.get("config") if isinstance(prev, dict) else None
            cfg_match = (
                isinstance(prev_cfg, dict)
                and prev_cfg.get("seed") == seed
                and prev_cfg.get("debug_mode") == debug_mode
                and prev_cfg.get("num_agents") == num_agents
                and prev_cfg.get("misinformed_count") == misinformed_count
                and prev_cfg.get("model_name") == model_name
                and prev_cfg.get("endpoint_url") == endpoint_url
            )
            total_match = (expected_total is None) or (prev_total == expected_total)
            if cfg_match and total_match:
                print(f"[CONTINUE] Skipping already-completed mix: {mix_key} (total_count={prev_total})")
                continue
            print(f"[CONTINUE] Re-running mix (incomplete or config changed): {mix_key}")

        input_path = os.path.join(mallm_io_dir, f"exp3_mallm_input_winogrande_{mix_key}.json")
        output_path = os.path.join(mallm_io_dir, f"exp3_mallm_output_winogrande_{mix_key}.json")

        _convert_winogrande_for_mix(
            src_path=src,
            dst_path=input_path,
            misinformed_count=misinformed_count,
            num_agents=num_agents,
            debug_mode=debug_mode,
            seed=seed,
        )

        cfg = Config(
            input_json_file_path=input_path,
            output_json_file_path=output_path,
            task_instruction_prompt="",
            task_instruction_prompt_template="winogrande",
            endpoint_url=endpoint_url,
            model_name=model_name,
            discussion_paradigm="memory",
            decision_protocol="simple_voting",
            max_turns=5,
            num_agents=num_agents,
            agent_generator="informed",
            agent_generators_list=["informed"] * num_agents,
            use_chain_of_thought=False,
            concurrent_api_requests=250,
            shuffle_input_samples=False,
        )

        mallm_scheduler = scheduler.Scheduler(cfg)
        mallm_scheduler.run()

        mix_results = _evaluate_and_aggregate(output_path)
        mix_results["config"] = {
            "dataset": "winogrande",
            "num_agents": num_agents,
            "misinformed_count": misinformed_count,
            "informed_count": informed_count,
            "seed": seed,
            "debug_mode": debug_mode,
            "expected_total_count": expected_total,
            "endpoint_url": endpoint_url,
            "model_name": model_name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        _save_aggregate(mix_key, mix_results)
        print(f"MALLM debate complete for winogrande [{mix_key}]. Output: {output_path}")


def _env_or_default_endpoint(default_url: str) -> str:
    """
    Resolve endpoint from environment variables, falling back to the provided default.
    Supported env vars: MALLM_ENDPOINT_URL, OPENAI_BASE_URL, OPENAI_API_BASE
    """
    return (
        os.environ.get("MALLM_ENDPOINT_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or default_url
    )


def _endpoint_alive(endpoint_url: str, timeout: float = 2.0) -> bool:
    """
    Lightweight connectivity probe. Returns True if TCP/HTTP responds (any non-5xx).
    Assumes endpoint_url already contains '/v1'.
    """
    try:
        import requests

        base = endpoint_url.rstrip("/")
        resp = requests.get(f"{base}/models", timeout=timeout)
        return resp.status_code < 500
    except Exception:
        return False


def _discover_endpoint_from_mallm(model_name: str) -> Optional[str]:
    """
    Try to discover a running mallm model endpoint by parsing mallm port files.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    mallm_repo_root = os.path.abspath(os.path.join(this_dir, "..", "mallm"))
    safe_model = model_name.replace("/", "-")
    pattern = os.path.join(mallm_repo_root, f"port-{safe_model}-*.txt")
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    for path in candidates:
        try:
            with open(path, "r") as f:
                text = f.read()
            m = re.search(
                r"Connect via:\s*ssh\s+-L\s+\d+:(?P<host>[A-Za-z0-9._-]+):(?P<port>\d+)",
                text,
            )
            if m:
                host = m.group("host")
                port = m.group("port")
                url = f"http://{host}:{port}/v1"
                if _endpoint_alive(url, timeout=1.5):
                    return url
            m_host = re.search(r"Running on instance:\s*(?P<host>\S+)", text)
            m_port = re.search(r"Port:\s*(?P<port>\d+)", text)
            if m_host and m_port:
                url = f"http://{m_host.group('host')}:{m_port.group('port')}/v1"
                if _endpoint_alive(url, timeout=1.5):
                    return url
        except Exception:
            continue
    return None


def _get_num_gpus() -> int:
    try:
        import torch  # type: ignore

        n = int(torch.cuda.device_count())
        if n > 0:
            return n
    except Exception:
        pass
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        try:
            return len([x for x in cvd.split(",") if x.strip() != ""])
        except Exception:
            return 1
    return 1


def _port_is_busy(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            return s.connect_ex((host, port)) == 0
        except Exception:
            return False


def _find_free_port(start_port: int = 8080, host: str = "127.0.0.1", max_tries: int = 50) -> int:
    port = start_port
    for _ in range(max_tries):
        if not _port_is_busy(host, port):
            return port
        port += 1
    return start_port


def _model_server_backend_from_env() -> str:
    for key in ("MODEL_SERVER_BACKEND", "VLLM_SERVER_BACKEND"):
        raw = (os.environ.get(key) or "").strip().lower()
        if raw:
            if raw in ("apptainer", "venv", "hf_openai_shim"):
                return "auto"
            return raw
    return "auto"


def _build_sglang_server_command(model_name: str, host: str, port: int, num_gpus: int) -> Optional[list[str]]:
    py_override = (os.environ.get("SGLANG_SERVER_PYTHON") or "").strip()
    exe = py_override if py_override and os.path.isfile(py_override) else sys.executable
    try:
        chk = subprocess.run(
            [exe, "-c", "import sglang"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if chk.returncode != 0:
            return None
    except Exception:
        return None

    cmd: list[str] = [
        exe,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--tp",
        str(max(1, num_gpus)),
    ]
    trc_env = (os.environ.get("SGLANG_TRUST_REMOTE_CODE") or "").strip().lower()
    trust = is_glm_model(model_name) or trc_env in ("1", "true", "yes")
    if trust:
        cmd.append("--trust-remote-code")
    mf = (os.environ.get("SGLANG_MEM_FRACTION_STATIC") or "").strip()
    if mf:
        cmd += ["--mem-fraction-static", mf]
    mrr = (os.environ.get("SGLANG_MAX_RUNNING_REQUESTS") or "").strip()
    if mrr:
        cmd += ["--max-running-requests", mrr]
    ctx = (os.environ.get("SGLANG_CONTEXT_LENGTH") or "").strip()
    if ctx:
        cmd += ["--context-length", ctx]
    cps = (os.environ.get("SGLANG_CHUNKED_PREFILL_SIZE") or "").strip()
    if cps:
        cmd += ["--chunked-prefill-size", cps]
    tcp = (os.environ.get("SGLANG_TOOL_CALL_PARSER") or "").strip()
    if not tcp:
        tcp = sglang_tool_call_parser_default(model_name)
    if tcp:
        cmd += ["--tool-call-parser", tcp]
    rp = (os.environ.get("SGLANG_REASONING_PARSER") or "").strip()
    if rp:
        cmd += ["--reasoning-parser", rp]
    extra = (os.environ.get("SGLANG_EXTRA_ARGS") or "").strip()
    if extra:
        cmd.extend(shlex.split(extra))
    return cmd


def _start_model_server(model_name: str, desired_port: Optional[int] = None) -> Optional[dict]:
    """
    Attempt to start a local OpenAI-compatible model server (aligned with exp2.py).
    """
    host = "127.0.0.1"
    port = desired_port or 8080
    if _port_is_busy(host, port):
        port = _find_free_port(start_port=port, host=host)

    num_gpus = max(1, _get_num_gpus())
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(logs_dir, exist_ok=True)

    launcher = shutil.which("text-generation-launcher")
    proc: Optional[subprocess.Popen] = None
    server_kind: Optional[str] = None
    log_path: Optional[str] = None
    f = None
    backend = _model_server_backend_from_env()

    def _start_tgi() -> bool:
        nonlocal proc, server_kind, log_path, f
        if launcher is None:
            return False
        max_conc = os.environ.get("TGI_MAX_CONCURRENT_REQUESTS", "280").strip() or "280"
        cmd = [
            launcher,
            "--model-id",
            model_name,
            "--port",
            str(port),
            "--num-shard",
            str(num_gpus),
            "--hostname",
            host,
            "--max-concurrent-requests",
            str(max_conc),
        ]
        if is_glm_model(model_name) or os.environ.get("TGI_TRUST_REMOTE_CODE", "").strip() in (
            "1",
            "true",
            "yes",
        ):
            cmd.append("--trust-remote-code")
        max_total = os.environ.get("TGI_MAX_TOTAL_TOKENS", "").strip()
        if max_total:
            cmd += ["--max-total-tokens", max_total]
        max_batch = os.environ.get("TGI_MAX_BATCH_TOTAL_TOKENS", "").strip()
        if max_batch:
            cmd += ["--max-batch-total-tokens", max_batch]
        log_path = os.path.join(logs_dir, f"tgi_server_{port}.log")
        f = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        server_kind = "tgi"
        return True

    def _start_sglang() -> bool:
        nonlocal proc, server_kind, log_path, f
        cmd = _build_sglang_server_command(model_name, host, port, num_gpus)
        if not cmd:
            return False
        log_path = os.path.join(logs_dir, f"sglang_server_{port}.log")
        f = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        server_kind = "sglang"
        return True

    def _start_vllm() -> bool:
        nonlocal proc, server_kind, log_path, f
        vllm_module = "vllm.entrypoints.openai.api_server"
        try:
            apply_gpt_oss_tiktoken_env(model_name)
            __import__("vllm")
            _mml = os.environ.get("VLLM_MAX_MODEL_LEN", "").strip()
            if _mml:
                max_model_len = int(_mml)
            else:
                max_model_len = vllm_max_model_len_default(model_name)
            gpu_mem_util = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "").strip()
            if not gpu_mem_util:
                gpu_mem_util = str(vllm_gpu_memory_utilization_default(model_name))
            kv_cache_dtype = os.environ.get("VLLM_KV_CACHE_DTYPE", "").strip()
            max_num_batched_tokens = os.environ.get("VLLM_MAX_BATCHED_TOKENS", "").strip()

            cmd = [
                sys.executable,
                "-m",
                vllm_module,
                "--host",
                host,
                "--port",
                str(port),
                "--model",
                model_name,
                "--tensor-parallel-size",
                str(num_gpus),
            ]
            trc = (os.environ.get("VLLM_TRUST_REMOTE_CODE") or "").strip().lower()
            if not trc and vllm_trust_remote_code_default(model_name):
                cmd.append("--trust-remote-code")
            elif trc in ("1", "true", "yes"):
                cmd.append("--trust-remote-code")
            cmd += ["--max-model-len", str(max_model_len)]
            if gpu_mem_util:
                cmd += ["--gpu-memory-utilization", str(gpu_mem_util)]
            if kv_cache_dtype:
                cmd += ["--kv-cache-dtype", kv_cache_dtype]
            if max_num_batched_tokens:
                cmd += ["--max-num-batched-tokens", str(max_num_batched_tokens)]
            cmd.extend(vllm_extra_args_for_model(model_name))
            log_path = os.path.join(logs_dir, f"vllm_server_{port}.log")
            f = open(log_path, "w")
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
            server_kind = "vllm"
            return True
        except Exception:
            proc = None
            server_kind = None
            return False

    try:
        if backend == "sglang":
            if not _start_sglang():
                proc = None
                server_kind = None
        elif backend == "vllm":
            if not _start_vllm():
                proc = None
                server_kind = None
        elif backend == "tgi":
            if not _start_tgi():
                proc = None
                server_kind = None
        else:
            if launcher is not None:
                _start_tgi()
            elif not _start_vllm():
                proc = None
                server_kind = None
    except Exception:
        proc = None
        server_kind = None

    if proc is None or server_kind is None:
        if log_path:
            try:
                f.close()  # type: ignore[name-defined]
            except Exception:
                pass
        return None

    def _terminate_server() -> None:
        try:
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass

    atexit.register(_terminate_server)

    endpoint = f"http://{host}:{port}/v1"
    print(
        f"[INFO] Started {server_kind} server on {host}:{port} for model '{model_name}'. Waiting for readiness..."
    )
    start_t = time.time()
    max_wait_s = int(os.environ.get("MODEL_STARTUP_TIMEOUT_SECONDS", "1800"))
    probe_interval = 5.0
    last_report = -1
    while time.time() - start_t < max_wait_s:
        if _endpoint_alive(endpoint, timeout=2.5):
            print(f"[INFO] Model server is ready at {endpoint}.")
            return {"process": proc, "endpoint_url": endpoint, "kind": server_kind, "port": port}
        elapsed = int(time.time() - start_t)
        if elapsed // 15 != last_report // 15:
            print(f"[INFO] Waiting for model server... {elapsed}s elapsed")
            last_report = elapsed
        if proc.poll() is not None:
            print(f"[ERROR] Model server process exited prematurely. See log: {log_path}")
            break
        time.sleep(probe_interval)

    print(f"[ERROR] Timed out waiting for model server to start after {max_wait_s}s. See log: {log_path}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-agent debate (MALLM) on winogrande with varying misinformed/informed agent composition."
    )
    parser.add_argument("--mock", action="store_true", help="Use mock:// endpoint with lightweight stubs for a quick test run.")
    parser.add_argument("--endpoint_url", type=str, default="http://127.0.0.1:8080/v1", help="OpenAI-compatible endpoint URL for real runs.")
    parser.add_argument(
        "--model_name",
        "--model",
        dest="model_name",
        type=str,
        default="openai/gpt-oss-120b",
        help="Model name for the endpoint (e.g. zai-org/GLM-4.7-Flash).",
    )
    parser.add_argument("--no_autostart", action="store_true", help="Do not try to auto-start a local model server if endpoint is unreachable.")
    parser.add_argument("--debug", action="store_true", help="Debug mode: process only 5 samples per mix.")
    parser.add_argument(
        "--continue",
        dest="continue_mode",
        action="store_true",
        help=(
            "Resume: skip mixes already completed in "
            "out/<model_name>/exp3_results_winogrande.json "
            "(only safe if args match)."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used for deterministic per-example shuffling of misinformed slots.")
    args = parser.parse_args()
    resolved_model_name = normalize_model_name(args.model_name)

    if args.mock:
        run_with_mallm(True, "mock://local", resolved_model_name, debug_mode=args.debug, seed=args.seed, continue_mode=args.continue_mode)
        return

    resolved_endpoint = _env_or_default_endpoint(args.endpoint_url)
    if resolved_endpoint.startswith("http") and not _endpoint_alive(resolved_endpoint, timeout=1.5):
        auto = _discover_endpoint_from_mallm(resolved_model_name)
        if auto:
            print(f"[INFO] Provided endpoint '{resolved_endpoint}' is unreachable. Using discovered endpoint '{auto}' instead.")
            resolved_endpoint = auto
        else:
            print(f"[WARN] Endpoint '{resolved_endpoint}' is unreachable.")
            if not args.no_autostart:
                try:
                    host_port = re.search(r"https?://([^:/]+):(\d+)/v1", resolved_endpoint)
                    desired_port = 8080
                    if host_port and host_port.group(1) in ("127.0.0.1", "localhost"):
                        try:
                            desired_port = int(host_port.group(2))
                        except Exception:
                            desired_port = 8080
                    started = _start_model_server(resolved_model_name, desired_port=desired_port)
                    if started:
                        resolved_endpoint = started["endpoint_url"]  # type: ignore[index]
                    else:
                        print("[ERROR] Could not auto-start a model server. Please start one manually or provide a reachable --endpoint_url.")
                except Exception as e:
                    print(f"[ERROR] Auto-start failed: {e}. Please start a model server or set --no_autostart.")
            else:
                print("[INFO] Auto-start disabled (--no_autostart). Proceeding without starting a local server.")

    run_with_mallm(False, resolved_endpoint, resolved_model_name, debug_mode=args.debug, seed=args.seed, continue_mode=args.continue_mode)


if __name__ == "__main__":
    main()


