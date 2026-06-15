#!/usr/bin/env python3

"""
Variant of exp1.py using irrelevant_true_information as extra context.
The prompt wording remains identical to exp1's misinformed prompt.

Inference backends:
  - hf: local HuggingFace `transformers` + `generate()` (original behavior).
  - openai: OpenAI-compatible HTTP API (vLLM, TGI, SGLang, etc.), aligned with exp2/exp3 Slurm jobs.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from exp1 import process_batch_responses
from model_server_config import (
    extract_openai_message_text,
    mallm_concurrency_default,
    normalize_model_name,
)
from shared_utils import (
    DummyModel,
    DummyTokenizer,
    PROMPT_BASELINE,
    PROMPT_BASELINE_FREE_FORM,
    PROMPT_MISINFORMED,
    PROMPT_MISINFORMED_FREE_FORM,
    build_common_arg_parser,
    is_multiple_choice_dataset,
    load_dataset,
    format_options,
    model_name_to_safe_path,
    save_results,
)


def prepare_batch_prompts(batch_items: list, prompt_template: str) -> Tuple[list, list]:
    """Prepare prompts for a batch of items. Returns (prompts, info_strategies)."""
    prompts = []
    info_strategies = []

    for item in batch_items:
        is_multiple_choice = is_multiple_choice_dataset(item)
        sentence = item["sentence"]
        options_text = format_options(item["options"]) if is_multiple_choice else ""

        if "misinformation" in prompt_template:
            extra_information = item.get("irrelevant_true_information")
            if not isinstance(extra_information, str) or not extra_information.strip():
                raise ValueError(
                    "Missing 'irrelevant_true_information' in dataset item. "
                    "Use MINT-dataset_v1.1 files for exp1a."
                )

            if is_multiple_choice:
                prompt = PROMPT_MISINFORMED.format(
                    sentence=sentence,
                    options_text=options_text,
                    misinformation=extra_information,
                )
            else:
                prompt = PROMPT_MISINFORMED_FREE_FORM.format(
                    sentence=sentence,
                    misinformation=extra_information,
                )
            info_strategies.append("irrelevant_true_information")
        else:
            if is_multiple_choice:
                prompt = PROMPT_BASELINE.format(
                    sentence=sentence,
                    options_text=options_text,
                )
            else:
                prompt = PROMPT_BASELINE_FREE_FORM.format(sentence=sentence)
            info_strategies.append("none")

        prompts.append(prompt)

    return prompts, info_strategies


def _openai_timeout_seconds() -> float:
    raw = (os.environ.get("MALLM_OPENAI_TIMEOUT_SECONDS") or "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return 600.0


def _openai_chat_text(
    client: Any,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    max_retries: int = 4,
) -> str:
    delay = 1.0
    last_err: Optional[BaseException] = None
    for _attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            return extract_openai_message_text(resp.choices[0].message)
        except Exception as e:
            last_err = e
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    assert last_err is not None
    raise last_err


def evaluate_model(
    model: Any,
    tokenizer: Any,
    dataset: list,
    dataset_name: str,
    experimental_condition: str,
    max_samples: Optional[int] = None,
    batch_size: int = 8,
    max_tokens: int = 512,
    save_callback: Optional[Callable[..., Any]] = None,
    *,
    inference: str = "hf",
    openai_client: Optional[Any] = None,
    api_model_name: Optional[str] = None,
    concurrent_requests: int = 64,
    openai_timeout: Optional[float] = None,
) -> dict:
    """Evaluate model on a dataset with a specific experimental condition."""
    if max_samples:
        dataset = dataset[:max_samples]

    if experimental_condition == "baseline":
        prompt_template = PROMPT_BASELINE
    elif experimental_condition == "irrelevant_true_information":
        prompt_template = PROMPT_MISINFORMED
    else:
        raise ValueError(f"Unknown experimental condition: {experimental_condition}")

    if inference == "openai":
        return _evaluate_model_openai(
            dataset,
            dataset_name,
            experimental_condition,
            prompt_template,
            save_callback,
            openai_client=openai_client,
            api_model_name=api_model_name or "",
            concurrent_requests=max(1, concurrent_requests),
            max_tokens=max_tokens,
            openai_timeout=openai_timeout or _openai_timeout_seconds(),
        )

    num_batches = (len(dataset) + batch_size - 1) // batch_size
    print(f"Evaluating {len(dataset)} samples for {dataset_name} with {experimental_condition} condition")
    print(f"  Using batch_size={batch_size} ({num_batches} batches) [inference=hf]")

    all_results: list = []
    correct_count = 0

    for i in tqdm(
        range(0, len(dataset), batch_size),
        desc=f"Processing {experimental_condition} [batch {batch_size}]",
        total=num_batches,
    ):
        batch_items = dataset[i : i + batch_size]
        prompts, info_strategies = prepare_batch_prompts(batch_items, prompt_template)

        if isinstance(model, DummyModel):
            responses = []
            for prompt in prompts:
                is_free_form = "Provide a clear and accurate answer" in prompt or "Always provide your final answer choice" not in prompt
                if is_free_form:
                    response = random.choice(
                        [
                            "Real Madrid C.F.",
                            "Remember the Titans",
                            "The Twilight Saga: New Moon",
                            "Mysophobia",
                            "Albanian language, Greek Language",
                            "Vancouver",
                            "The answer is Real Madrid.",
                            "Based on the information, it's Remember the Titans.",
                            "The correct answer is Vancouver.",
                        ]
                    )
                else:
                    response = random.choice(
                        [
                            "Based on my analysis, the answer is A) First option.",
                            "After careful consideration, I believe the correct answer is B) Second option.",
                            "Looking at the evidence, the answer appears to be C) Third option.",
                        ]
                    )
                responses.append(response)
        else:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )

            responses = []
            for j, output in enumerate(outputs):
                input_length = inputs["input_ids"][j].shape[0]
                generated_tokens = output[input_length:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                responses.append(generated_text)

        batch_results = process_batch_responses(
            responses, batch_items, dataset_name, prompts, info_strategies
        )
        all_results.extend(batch_results)
        correct_count += sum(1 for result in batch_results if result["is_correct"])

        if save_callback:
            current_accuracy = correct_count / len(all_results) if all_results else 0.0
            save_callback(
                experimental_condition,
                {
                    "accuracy": current_accuracy,
                    "correct_count": correct_count,
                    "total_count": len(all_results),
                    "results": all_results.copy(),
                },
            )

    accuracy = correct_count / len(all_results) if all_results else 0.0
    print(f"{experimental_condition} accuracy: {accuracy:.3f} ({correct_count}/{len(all_results)})")
    return {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": len(all_results),
        "results": all_results,
    }


def _evaluate_model_openai(
    dataset: list,
    dataset_name: str,
    experimental_condition: str,
    prompt_template: str,
    save_callback: Optional[Callable[..., Any]],
    *,
    openai_client: Any,
    api_model_name: str,
    concurrent_requests: int,
    max_tokens: int,
    openai_timeout: float,
) -> dict:
    """Parallel OpenAI-compatible chat completions over sliding windows (like exp2 concurrency)."""
    print(f"Evaluating {len(dataset)} samples for {dataset_name} with {experimental_condition} condition")
    print(
        f"  Using inference=openai, concurrent_requests={concurrent_requests}, "
        f"api_model_name={api_model_name!r}"
    )

    window = max(1, concurrent_requests)
    num_windows = (len(dataset) + window - 1) // window
    temperature = 0.7

    all_results: list = []
    correct_count = 0

    for w in tqdm(range(num_windows), desc=f"OpenAI {experimental_condition} [win={window}]", total=num_windows):
        start = w * window
        chunk = dataset[start : start + window]
        prompts, info_strategies = prepare_batch_prompts(chunk, prompt_template)

        def _one(ix: int, prompt: str) -> Tuple[int, str]:
            text = _openai_chat_text(
                openai_client,
                api_model_name,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=openai_timeout,
            )
            return ix, text

        max_workers = min(len(prompts), concurrent_requests)
        responses_ordered: List[Optional[str]] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
            futures = {ex.submit(_one, j, p): j for j, p in enumerate(prompts)}
            for fut in as_completed(futures):
                j, text = fut.result()
                responses_ordered[j] = text

        responses = [r if r is not None else "" for r in responses_ordered]
        batch_results = process_batch_responses(
            responses, chunk, dataset_name, prompts, info_strategies
        )
        all_results.extend(batch_results)
        correct_count += sum(1 for result in batch_results if result["is_correct"])

        if save_callback:
            current_accuracy = correct_count / len(all_results) if all_results else 0.0
            save_callback(
                experimental_condition,
                {
                    "accuracy": current_accuracy,
                    "correct_count": correct_count,
                    "total_count": len(all_results),
                    "results": all_results.copy(),
                },
            )

    accuracy = correct_count / len(all_results) if all_results else 0.0
    print(f"{experimental_condition} accuracy: {accuracy:.3f} ({correct_count}/{len(all_results)})")
    return {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": len(all_results),
        "results": all_results,
    }


def run_experiment(
    dataset_name: str,
    dataset: list,
    model: Any,
    tokenizer: Any,
    max_samples: Optional[int] = None,
    batch_size: int = 8,
    max_tokens: int = 512,
    save_callback: Optional[Callable[..., Any]] = None,
    conditions: Tuple[str, ...] = ("irrelevant_true_information",),
    *,
    inference: str = "hf",
    openai_client: Optional[Any] = None,
    api_model_name: Optional[str] = None,
    concurrent_requests: int = 64,
    openai_timeout: Optional[float] = None,
) -> dict:
    """Run selected experimental conditions (default: irrelevant_true_information only)."""
    print(f"\n=== Running experiment for {dataset_name} ===")
    allowed = {"baseline", "irrelevant_true_information"}
    unknown = set(conditions) - allowed
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}. Allowed: {sorted(allowed)}")
    if not conditions:
        raise ValueError("At least one condition is required.")

    results = {}
    for condition in ("baseline", "irrelevant_true_information"):
        if condition not in conditions:
            continue
        results[condition] = evaluate_model(
            model,
            tokenizer,
            dataset,
            dataset_name,
            condition,
            max_samples,
            batch_size,
            max_tokens,
            save_callback,
            inference=inference,
            openai_client=openai_client,
            api_model_name=api_model_name,
            concurrent_requests=concurrent_requests,
            openai_timeout=openai_timeout,
        )
    return results


def _normalize_dataset_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    key = name.strip().lower().replace("-", "_")
    aliases = {
        "ethics_commonsense": "ethics",
        "ethics": "ethics",
        "wino": "winogrande",
        "winogrande": "winogrande",
        "complex": "complex_web_questions",
        "cwq": "complex_web_questions",
        "complex_web_questions": "complex_web_questions",
        "web_questions": "complex_web_questions",
    }
    return aliases.get(key, key)


def _resolve_dataset_files() -> list:
    """Prefer v1.1 files that include irrelevant_true_information."""
    candidates = [
        "MINT-dataset_v1.1/winogrande_misinformed.json",
        "MINT-dataset_v1.1/ethics_commonsense_misinformed.json",
        "MINT-dataset_v1.1/complex_web_questions_misinformed.json",
        # Fallbacks if a user copied v1.1 files into data/.
        "data/winogrande_misinformed.json",
        "data/ethics_commonsense_misinformed.json",
        "data/complex_web_questions_misinformed.json",
    ]
    # Keep first existing file per dataset stem.
    seen = set()
    resolved = []
    for fp in candidates:
        stem = os.path.basename(fp)
        if stem in seen:
            continue
        if os.path.exists(fp):
            seen.add(stem)
            resolved.append(fp)
    return resolved


def _normalize_item_schema(item: dict) -> dict:
    """Normalize MINT v1.1 free-form schema to exp1/shared_utils expectations."""
    if not isinstance(item, dict):
        return item

    # exp1/shared_utils expects free-form answers at item["answers"]["answer"].
    if "answers" not in item:
        answer_obj = item.get("answer")
        if isinstance(answer_obj, dict) and "answer" in answer_obj:
            item["answers"] = {"answer": answer_obj.get("answer", [])}
    return item


def _parse_conditions(conditions_str: str) -> Tuple[str, ...]:
    parts = [p.strip() for p in conditions_str.split(",") if p.strip()]
    if not parts:
        raise ValueError("--conditions must list at least one of: baseline, irrelevant_true_information")
    return tuple(parts)


def _resolve_openai_endpoint(args: argparse.Namespace, resolved_model_name: str) -> str:
    from exp2 import (
        _discover_endpoint_from_mallm,
        _endpoint_alive,
        _env_or_default_endpoint,
        _start_model_server,
    )

    resolved_endpoint = _env_or_default_endpoint(args.endpoint_url)
    if resolved_endpoint.startswith("http") and not _endpoint_alive(resolved_endpoint, timeout=2.0):
        auto = _discover_endpoint_from_mallm(resolved_model_name)
        if auto:
            print(f"[INFO] Endpoint '{args.endpoint_url}' unreachable; using discovered endpoint '{auto}'.")
            resolved_endpoint = auto
        elif not args.no_autostart:
            print(f"[WARN] Endpoint '{resolved_endpoint}' unreachable; attempting local model server autostart.")
            desired_port = 8080
            m = re.search(r"https?://([^:/]+):(\d+)/v1", resolved_endpoint)
            if m and m.group(1) in ("127.0.0.1", "localhost"):
                try:
                    desired_port = int(m.group(2))
                except ValueError:
                    desired_port = 8080
            started = _start_model_server(resolved_model_name, desired_port=desired_port)
            if started:
                resolved_endpoint = started["endpoint_url"]
            else:
                print("[ERROR] Could not auto-start a model server. Start vLLM/TGI manually or pass --endpoint_url.")
        else:
            print("[INFO] Auto-start disabled (--no_autostart).")

    if resolved_endpoint.startswith("http") and not _endpoint_alive(resolved_endpoint, timeout=2.5):
        raise SystemExit(
            f"[FATAL] No reachable model endpoint at {resolved_endpoint!r}. "
            "For Slurm, launch vLLM first (see exp1a.slurm) or pass a live --endpoint_url."
        )
    return resolved_endpoint


def main() -> None:
    parser = build_common_arg_parser("Run single agent experiment with irrelevant true information")
    parser.add_argument(
        "--conditions",
        type=str,
        default="irrelevant_true_information",
        help=(
            "Comma-separated conditions to run: baseline, irrelevant_true_information. "
            "Default runs irrelevant_true_information only. "
            "Use 'baseline,irrelevant_true_information' for the full exp1a comparison."
        ),
    )
    parser.add_argument(
        "--inference",
        choices=("hf", "openai"),
        default=os.environ.get("EXP1A_INFERENCE", "hf"),
        help="hf: local transformers. openai: vLLM/TGI OpenAI-compatible API (see exp2).",
    )
    parser.add_argument(
        "--endpoint_url",
        type=str,
        default="http://127.0.0.1:8080/v1",
        help="OpenAI-compatible base URL including /v1 (used when --inference openai).",
    )
    parser.add_argument(
        "--no_autostart",
        action="store_true",
        help="Do not try to start a local vLLM/TGI process if the endpoint is down (Slurm sets this).",
    )
    parser.add_argument(
        "--concurrent_requests",
        type=int,
        default=None,
        help="Max parallel chat completions for openai inference. "
        "Default: env MALLM_CONCURRENT_API_REQUESTS or 64.",
    )
    args = parser.parse_args()

    print("=== Single Agent Supplementary Experiment (Irrelevant True Information) ===")
    print(f"Model: {args.model_name}")
    print(f"Inference: {args.inference}")
    print(f"Max samples per dataset: {args.max_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Dummy test mode: {args.dummy_test}")
    if getattr(args, "datasets", None):
        print(f"Selected datasets: {args.datasets}")
    conditions = _parse_conditions(args.conditions)
    print(f"Conditions: {', '.join(conditions)}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if args.use_smaller_model:
        resolved_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    else:
        resolved_model_name = normalize_model_name(args.model_name)
    concurrent_requests = (
        args.concurrent_requests
        if args.concurrent_requests is not None
        else mallm_concurrency_default(resolved_model_name)
    )
    openai_timeout = _openai_timeout_seconds()

    openai_client: Optional[Any] = None
    if args.dummy_test:
        model = DummyModel()
        tokenizer = DummyTokenizer()
    elif args.inference == "openai":
        resolved_endpoint = _resolve_openai_endpoint(args, resolved_model_name)
        from openai import OpenAI

        openai_client = OpenAI(
            base_url=resolved_endpoint.rstrip("/"),
            api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
            max_retries=int(os.environ.get("MALLM_OPENAI_MAX_RETRIES", "5")),
        )
        model = None
        tokenizer = None
        print(f"[INFO] OpenAI client base_url={resolved_endpoint!r} model_id={resolved_model_name!r}")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_kwargs: dict = {}
        if args.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        elif args.load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name,
            **({"torch_dtype": torch.float16} if not (args.load_in_4bit or args.load_in_8bit) else {}),
            device_map="auto",
            trust_remote_code=True,
            **model_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    datasets = {}
    selected_datasets = None
    if getattr(args, "datasets", None):
        selected_datasets = {
            _normalize_dataset_name(x)
            for x in str(args.datasets).split(",")
            if str(x).strip()
        }

    for file_path in _resolve_dataset_files():
        dataset_name = os.path.basename(file_path).replace("_misinformed.json", "")
        dataset_name = "ethics" if dataset_name == "ethics_commonsense" else dataset_name
        dataset_name = _normalize_dataset_name(dataset_name)

        if selected_datasets is not None and dataset_name not in selected_datasets:
            print(f"Skipping dataset {dataset_name} (not selected)")
            continue

        ds = load_dataset(file_path)
        ds = [_normalize_item_schema(x) for x in ds]
        if not ds:
            continue
        if "irrelevant_true_information" not in ds[0]:
            raise ValueError(
                f"{file_path} is missing 'irrelevant_true_information'. "
                "Please use MINT-dataset_v1.1 files."
            )
        datasets[dataset_name] = ds
        print(f"Loaded {len(ds)} samples from {dataset_name} ({file_path})")

    if not datasets:
        raise ValueError("No suitable datasets found with 'irrelevant_true_information'.")

    model_name_suffix = "dummy_model" if args.dummy_test else model_name_to_safe_path(resolved_model_name)
    os.makedirs("out", exist_ok=True)

    all_results = {}
    for dataset_name, dataset in datasets.items():
        output_file = f"out/exp1a_results_{dataset_name}_{model_name_suffix}.json"

        def create_save_callback(target_output_file: str):
            def save_callback(condition: str, condition_results: dict) -> None:
                if os.path.exists(target_output_file):
                    try:
                        with open(target_output_file, "r", encoding="utf-8") as f:
                            existing_results = json.load(f)
                    except (json.JSONDecodeError, FileNotFoundError):
                        existing_results = {}
                else:
                    existing_results = {}
                existing_results[condition] = condition_results
                save_results(existing_results, target_output_file)
                print(f"  Saved {condition} results to {target_output_file} (accuracy: {condition_results['accuracy']:.3f})")

            return save_callback

        results = run_experiment(
            dataset_name,
            dataset,
            model,
            tokenizer,
            args.max_samples,
            args.batch_size,
            args.max_tokens,
            create_save_callback(output_file),
            conditions=conditions,
            inference=args.inference,
            openai_client=openai_client,
            api_model_name=resolved_model_name,
            concurrent_requests=concurrent_requests,
            openai_timeout=openai_timeout,
        )
        all_results[dataset_name] = results

    print("\n=== Experiment Summary ===")
    for dataset_name, dataset_results in all_results.items():
        print(f"\n{dataset_name}:")
        for condition, condition_results in dataset_results.items():
            print(
                f"  {condition}: {condition_results['accuracy']:.3f} "
                f"({condition_results['correct_count']}/{condition_results['total_count']})"
            )


if __name__ == "__main__":
    main()
