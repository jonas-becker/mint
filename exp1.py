#!/usr/bin/env python3

"""
Refactored exp1.py - Single Agent Experiment with Misinformation
This version uses shared utilities to reduce code duplication.
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_server_config import (
    extract_openai_message_text,
    mallm_concurrency_default,
    normalize_model_name,
)

# Import shared utilities
from shared_utils import (
    build_common_arg_parser,
    load_dataset,
    format_options,
    extract_answer_from_response,
    is_multiple_choice_dataset, get_correct_answers, evaluate_answer,
    save_results, DummyModel, DummyTokenizer, model_name_to_safe_path,
    PROMPT_BASELINE, PROMPT_MISINFORMED, PROMPT_BASELINE_FREE_FORM, PROMPT_MISINFORMED_FREE_FORM
)

def prepare_batch_prompts(batch_items: list, prompt_template: str, dataset: list = None, irrelevant: bool = False, condition: str = None) -> tuple:
    """Prepare prompts for a batch of items. Returns (prompts, misinformation_strategies).
    
    Supports:
    - baseline (no misinformation)
    - false_fact (use item['false_fact'])
    - strategy_<name> (use item['misinformation_by_strategy'][<name>])
    - irrelevant_misinformed (pick misinformation from other items)
    - misinformed (alias to false_fact for backward compatibility)
    """
    prompts = []
    misinformation_strategies = []
    
    for item in batch_items:
        # Determine if this is a multiple choice or free-form question
        is_multiple_choice = is_multiple_choice_dataset(item)
        
        if is_multiple_choice:
            # For winogrande and ethics datasets (multiple choice)
            options = item["options"]
            sentence = item["sentence"]
            options_text = format_options(options)
        else:
            # For complex_web_questions dataset (free-form)
            sentence = item["sentence"]
            options_text = ""
        
        # Choose appropriate prompt template and determine misinformation strategy
        misinformation_strategy = "none"
        
        # Determine misinformation based on condition
        if "misinformation" in prompt_template:
            if irrelevant:
                # Select a random misinformation from any other item in the dataset, using new fields
                misinformation_pool = []
                if dataset:
                    # Build a pool of misinfo from other items
                    for other in dataset:
                        if other is item:
                            continue
                        # false_fact
                        if isinstance(other, dict) and "false_fact" in other and isinstance(other["false_fact"], str):
                            misinformation_pool.append(("irrelevant_false_fact", other["false_fact"]))
                        # strategies
                        strat_map = other.get("misinformation_by_strategy") or {}
                        if isinstance(strat_map, dict):
                            for strat_name, strat_text in strat_map.items():
                                if isinstance(strat_text, str):
                                    misinformation_pool.append((f"irrelevant_{strat_name}", strat_text))
                if misinformation_pool:
                    misinformation_strategy, misinformation = random.choice(misinformation_pool)
                else:
                    misinformation = "This is irrelevant information."
                    misinformation_strategy = "irrelevant_misinformation"
            else:
                # Explicit conditions
                if condition == "false_fact" or condition == "misinformed":
                    # Use false_fact as the primary misinfo for new datasets
                    misinformation = item.get("false_fact") or "This is some additional information."
                    misinformation_strategy = "false_fact"
                elif isinstance(condition, str) and condition.startswith("strategy_"):
                    strategy_name = condition[len("strategy_"):]
                    strat_map = item.get("misinformation_by_strategy") or {}
                    misinformation = strat_map.get(strategy_name)
                    if not isinstance(misinformation, str) or not misinformation.strip():
                        # Fallback if this item lacks the specified strategy
                        misinformation = item.get("false_fact") or "This is some additional information."
                    misinformation_strategy = strategy_name
                else:
                    # Backward compatibility: try legacy "misinformation" field or fallback
                    if "misinformation" in item and isinstance(item["misinformation"], str):
                        misinformation = item["misinformation"]
                        misinformation_strategy = "general_misinformation"
                    else:
                        # Prefer new fields if available
                        if "false_fact" in item and isinstance(item["false_fact"], str):
                            misinformation = item["false_fact"]
                            misinformation_strategy = "false_fact"
                        else:
                            strat_map = item.get("misinformation_by_strategy") or {}
                            if isinstance(strat_map, dict) and len(strat_map) > 0:
                                # Pick a deterministic strategy to keep runs stable
                                first_key = sorted([k for k in strat_map.keys() if isinstance(k, str)])[0]
                                misinformation = strat_map[first_key]
                                misinformation_strategy = first_key
                            else:
                                misinformation = "This is some additional information."
                                misinformation_strategy = "general_misinformation"
            
            # Use appropriate misinformed prompt
            if is_multiple_choice:
                prompt = PROMPT_MISINFORMED.format(
                    sentence=sentence,
                    options_text=options_text,
                    misinformation=misinformation
                )
            else:
                prompt = PROMPT_MISINFORMED_FREE_FORM.format(
                    sentence=sentence,
                    misinformation=misinformation
                )
        else:
            # Use appropriate baseline prompt
            if is_multiple_choice:
                prompt = PROMPT_BASELINE.format(
                    sentence=sentence,
                    options_text=options_text
                )
            else:
                prompt = PROMPT_BASELINE_FREE_FORM.format(
                    sentence=sentence
                )
        
        prompts.append(prompt)
        misinformation_strategies.append(misinformation_strategy)
    
    return prompts, misinformation_strategies

def process_batch_responses(responses: list, batch_items: list, dataset_name: str, prompts: list = None, misinformation_strategies: list = None) -> list:
    """Process batch responses and extract answers."""
    results = []
    
    for i, (response, item) in enumerate(zip(responses, batch_items)):
        # Get correct answers and determine question type
        correct_answers, correct_answer_text, is_multiple_choice = get_correct_answers(item, dataset_name)
        
        if is_multiple_choice:
            options = item["options"]
        else:
            options = None
        
        # Evaluate the response
        predicted_answer, is_correct = evaluate_answer(response, correct_answers, is_multiple_choice, options)
        
        result = {
            "sentence": item["sentence"],
            "prompt": prompts[i] if prompts else "",
            "response": response,
            "predicted_answer": predicted_answer,
            "correct_answer": correct_answer_text,
            "is_correct": is_correct,
            "is_multiple_choice": is_multiple_choice,
            "options": options if is_multiple_choice else correct_answers,  # For free-form, use correct answers
            "misinformation_strategy": misinformation_strategies[i] if misinformation_strategies else "none"
        }
        
        results.append(result)
    
    return results


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
            import re

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
            "For Slurm, launch vLLM first or pass a live --endpoint_url."
        )
    return resolved_endpoint


def evaluate_model(
    model,
    tokenizer,
    dataset: list,
    dataset_name: str,
    experimental_condition: str,
    max_samples: int = None,
    batch_size: int = 8,
    max_tokens: int = 512,
    save_callback: callable = None,
    *,
    inference: str = "hf",
    openai_client: Optional[Any] = None,
    api_model_name: Optional[str] = None,
    concurrent_requests: int = 64,
    openai_timeout: Optional[float] = None,
) -> dict:
    """Evaluate model on a dataset with a specific experimental condition."""
    
    # Limit samples if specified
    if max_samples:
        dataset = dataset[:max_samples]
    
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    print(f"Evaluating {len(dataset)} samples for {dataset_name} with {experimental_condition} condition")
    print(f"  Using batch_size={batch_size} ({num_batches} batches)")
    
    # Prepare prompts based on experimental condition
    if experimental_condition == "baseline":
        prompt_template = PROMPT_BASELINE
        dataset_for_prompts = None
        irrelevant = False
    elif experimental_condition in ["misinformed", "false_fact"] or (
        isinstance(experimental_condition, str) and experimental_condition.startswith("strategy_")
    ):
        # Use the misinformed prompt for false_fact and each strategy_* condition
        prompt_template = PROMPT_MISINFORMED
        dataset_for_prompts = dataset
        irrelevant = False
    elif experimental_condition == "irrelevant_misinformed":
        prompt_template = PROMPT_MISINFORMED
        dataset_for_prompts = dataset
        irrelevant = True
    else:
        raise ValueError(f"Unknown experimental condition: {experimental_condition}")

    if inference == "openai":
        window = max(1, concurrent_requests)
        num_windows = (len(dataset) + window - 1) // window
        print(f"  Using inference=openai, concurrent_requests={window}, api_model_name={api_model_name!r}")
        all_results = []
        correct_count = 0
        temperature = 0.7
        timeout = openai_timeout or _openai_timeout_seconds()
        for w in tqdm(
            range(num_windows),
            desc=f"OpenAI {experimental_condition} [win={window}]",
            total=num_windows,
        ):
            start = w * window
            chunk = dataset[start : start + window]
            prompts, misinformation_strategies = prepare_batch_prompts(
                chunk, prompt_template, dataset_for_prompts, irrelevant, condition=experimental_condition
            )

            def _one(ix: int, prompt: str) -> Tuple[int, str]:
                text = _openai_chat_text(
                    openai_client,
                    api_model_name or "",
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                )
                return ix, text

            responses_ordered: List[Optional[str]] = [None] * len(prompts)
            with ThreadPoolExecutor(max_workers=min(len(prompts), window)) as ex:
                futures = {ex.submit(_one, j, p): j for j, p in enumerate(prompts)}
                for fut in as_completed(futures):
                    j, text = fut.result()
                    responses_ordered[j] = text
            responses = [r if r is not None else "" for r in responses_ordered]
            batch_results = process_batch_responses(
                responses, chunk, dataset_name, prompts, misinformation_strategies
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

    all_results = []
    correct_count = 0

    # Process in batches
    num_batches = (len(dataset) + batch_size - 1) // batch_size  # Ceiling division
    batch_num = 0
    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Processing {experimental_condition} [batch {batch_size}]", total=num_batches):
        batch_num += 1
        batch_items = dataset[i:i + batch_size]
        
        # Prepare prompts for the batch
        prompts, misinformation_strategies = prepare_batch_prompts(
            batch_items, prompt_template, dataset_for_prompts, irrelevant, condition=experimental_condition
        )
        
        # Generate responses for the entire batch
        if isinstance(model, DummyModel):
            # Dummy mode - generate fake responses
            responses = []
            for j, prompt in enumerate(prompts):
                # Check if this is a complex_web_questions prompt (free-form)
                is_free_form = "Provide a clear and accurate answer" in prompt or "Always provide your final answer choice" not in prompt
                
                if is_free_form:
                    # For complex_web_questions, generate free-form responses
                    free_form_responses = [
                        "Real Madrid C.F.",
                        "Remember the Titans", 
                        "The Twilight Saga: New Moon",
                        "Mysophobia",
                        "Albanian language, Greek Language",
                        "Vancouver",
                        "The answer is Real Madrid.",
                        "Based on the information, it's Remember the Titans.",
                        "The correct answer is Vancouver.",
                        "This appears to be about Mysophobia.",
                        "The region speaks Albanian and Greek languages."
                    ]
                    response = random.choice(free_form_responses)
                else:
                    # For multiple choice questions, generate multiple choice responses
                    multiple_choice_responses = [
                        "Based on my analysis, the answer is A) First option.",
                        "After careful consideration, I believe the correct answer is B) Second option.",
                        "Looking at the evidence, the answer appears to be C) Third option."
                    ]
                    response = random.choice(multiple_choice_responses)
                
                responses.append(response)
        else:
            # Real model mode
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Extract only the newly generated tokens (excluding the input prompt)
            responses = []
            for i, output in enumerate(outputs):
                # Get the length of the input tokens
                input_length = inputs['input_ids'][i].shape[0]
                # Extract only the newly generated tokens
                generated_tokens = output[input_length:]
                # Decode only the generated part
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                responses.append(generated_text)
        
        # Process responses
        batch_results = process_batch_responses(responses, batch_items, dataset_name, prompts, misinformation_strategies)
        all_results.extend(batch_results)
        
        # Count correct answers
        correct_count += sum(1 for result in batch_results if result["is_correct"])
        
        # Save results incrementally after each batch
        if save_callback:
            current_accuracy = correct_count / len(all_results) if all_results else 0.0
            save_callback(experimental_condition, {
                "accuracy": current_accuracy,
                "correct_count": correct_count,
                "total_count": len(all_results),
                "results": all_results.copy()  # Make a copy to avoid reference issues
            })
        
        # Log batch progress
        if batch_num % 5 == 0 or batch_num == num_batches:  # Log every 5 batches and the last batch
            print(f"  Completed batch {batch_num}/{num_batches} ({len(all_results)}/{len(dataset)} samples processed)")
    
    # Calculate accuracy
    accuracy = correct_count / len(all_results) if all_results else 0.0
    
    print(f"{experimental_condition} accuracy: {accuracy:.3f} ({correct_count}/{len(all_results)})")
    print(f"  Processed {num_batches} batches with batch_size={batch_size}")
    
    return {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": len(all_results),
        "results": all_results
    }

def run_experiment(
    dataset_name: str,
    dataset: list,
    model,
    tokenizer,
    max_samples: int = None,
    batch_size: int = 8,
    max_tokens: int = 512,
    save_callback: callable = None,
    **eval_kwargs,
) -> dict:
    """Run the complete experiment for a dataset."""
    
    print(f"\n=== Running experiment for {dataset_name} ===")
    print(f"Dataset size: {len(dataset)} samples, Batch size: {batch_size}")
    print(f"Expected batches: {(len(dataset) + batch_size - 1) // batch_size}")
    
    results = {}
    
    # Always run baseline
    baseline_results = evaluate_model(
        model, tokenizer, dataset, dataset_name, "baseline",
        max_samples, batch_size, max_tokens, save_callback, **eval_kwargs
    )
    results["baseline"] = baseline_results
    
    # New: Run false_fact condition
    false_fact_results = evaluate_model(
        model, tokenizer, dataset, dataset_name, "false_fact",
        max_samples, batch_size, max_tokens, save_callback, **eval_kwargs
    )
    results["false_fact"] = false_fact_results
    
    # New: Enumerate and run all misinformation strategies present in the dataset
    unique_strategies = set()
    for item in dataset:
        strat_map = item.get("misinformation_by_strategy")
        if isinstance(strat_map, dict):
            for strat_name in strat_map.keys():
                if isinstance(strat_name, str) and strat_name.strip():
                    unique_strategies.add(strat_name.strip())
    unique_strategies = sorted(unique_strategies)
    
    for strategy_name in unique_strategies:
        condition_key = f"strategy_{strategy_name}"
        strategy_results = evaluate_model(
            model, tokenizer, dataset, dataset_name, condition_key,
            max_samples, batch_size, max_tokens, save_callback, **eval_kwargs
        )
        results[condition_key] = strategy_results
    
    # Keep irrelevant_misinformed for completeness if desired
    try:
        irrelevant_results = evaluate_model(
            model, tokenizer, dataset, dataset_name, "irrelevant_misinformed",
            max_samples, batch_size, max_tokens, save_callback, **eval_kwargs
        )
        results["irrelevant_misinformed"] = irrelevant_results
    except Exception as e:
        print(f"Skipping 'irrelevant_misinformed' due to error: {e}")
    
    return results

def main():
    """Main function to run the experiment."""
    parser = build_common_arg_parser("Run single agent experiment with misinformation")
    parser.add_argument(
        "--inference",
        choices=("hf", "openai"),
        default=os.environ.get("EXP1_INFERENCE", "hf"),
        help="hf: local transformers. openai: vLLM OpenAI-compatible API (see exp1.slurm).",
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
        help="Do not try to start a local vLLM process if the endpoint is down (Slurm sets this).",
    )
    parser.add_argument(
        "--concurrent_requests",
        type=int,
        default=None,
        help="Max parallel chat completions for openai inference.",
    )
    args = parser.parse_args()
    resolved_model_name = normalize_model_name(args.model_name)

    print("=== Single Agent Experiment with Misinformation ===")
    print(f"Model: {args.model_name}")
    print(f"Inference: {args.inference}")
    print(f"Max samples per dataset: {args.max_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Dummy test mode: {args.dummy_test}")
    if getattr(args, "datasets", None):
        print(f"Selected datasets: {args.datasets}")
    
    concurrent_requests = (
        args.concurrent_requests
        if args.concurrent_requests is not None
        else mallm_concurrency_default(resolved_model_name)
    )
    openai_timeout = _openai_timeout_seconds()
    openai_client: Optional[Any] = None

    # Load or initialize model and tokenizer
    if args.dummy_test:
        print("Using dummy model for testing")
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
        print(f"Loading model: {resolved_model_name}")

        model_kwargs = {}
        if args.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        elif args.load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        if args.use_smaller_model:
            load_model_name = "meta-llama/Llama-3.1-8B-Instruct"
        else:
            load_model_name = resolved_model_name

        model = AutoModelForCausalLM.from_pretrained(
            load_model_name,
            **({"torch_dtype": torch.float16} if not (args.load_in_4bit or args.load_in_8bit) else {}),
            device_map="auto",
            trust_remote_code=True,
            **model_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(load_model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    
    def normalize_dataset_name(name: str) -> str:
        """Normalize dataset name input to internal canonical keys."""
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
            "logiqa": "logiqa",
        }
        return aliases.get(key, key)

    # Load datasets
    datasets = {}
    dataset_files = [
        "data/winogrande_misinformed.json",
        "data/ethics_commonsense_misinformed.json",
        "data/complex_web_questions_misinformed.json",
        "data/logiqa_misinformed.json",
    ]

    selected_datasets = None
    if getattr(args, "datasets", None):
        selected_datasets = {
            normalize_dataset_name(x)
            for x in str(args.datasets).split(",")
            if str(x).strip()
        }
    
    for file_path in dataset_files:
            if os.path.exists(file_path):
                dataset_name = file_path.split("/")[-1].replace("_misinformed.json", "")
                # Map dataset names to preferred names
                if dataset_name == "ethics_commonsense":
                    dataset_name = "ethics"
                dataset_name = normalize_dataset_name(dataset_name)

                if selected_datasets is not None and dataset_name not in selected_datasets:
                    print(f"Skipping dataset {dataset_name} (not selected)")
                    continue
                datasets[dataset_name] = load_dataset(file_path)
                print(f"Loaded {len(datasets[dataset_name])} samples from {dataset_name}")
            else:
                print(f"Warning: Dataset file {file_path} not found")

    if selected_datasets is not None:
        # Validate user selection against available datasets that exist on disk
        available_selected = set(datasets.keys())
        missing = selected_datasets - available_selected
        if missing:
            available = sorted(set(["winogrande", "ethics", "complex_web_questions", "logiqa"]))
            raise ValueError(
                f"Unknown or unavailable dataset(s): {sorted(missing)}. "
                f"Available: {available}. "
                "Note: dataset files must exist in data/."
            )
    
    if not datasets:
        print("No datasets found! Please ensure dataset files exist in the data/ directory.")
        return
    
    # Set up output file
    model_name_suffix = "dummy_model" if args.dummy_test else model_name_to_safe_path(args.model_name)
    
    # Create output directory if it doesn't exist
    os.makedirs("out", exist_ok=True)
    
    # Run experiments
    all_results = {}
    
    for dataset_name, dataset in datasets.items():
        # Create output file for this dataset
        output_file = f"out/exp1_results_{dataset_name}_{model_name_suffix}.json"
        
        # Create save callback for incremental saving
        def create_save_callback(dataset_name, output_file):
            def save_callback(condition, condition_results):
                # Load existing results if file exists
                if os.path.exists(output_file):
                    try:
                        with open(output_file, 'r', encoding='utf-8') as f:
                            existing_results = json.load(f)
                    except (json.JSONDecodeError, FileNotFoundError):
                        existing_results = {}
                else:
                    existing_results = {}
                
                # Update with new condition results
                existing_results[condition] = condition_results
                
                # Save updated results
                save_results(existing_results, output_file)
                print(f"  Saved {condition} results to {output_file} (accuracy: {condition_results['accuracy']:.3f})")
            
            return save_callback
        
        save_callback = create_save_callback(dataset_name, output_file)
        
        results = run_experiment(
            dataset_name,
            dataset,
            model,
            tokenizer,
            args.max_samples,
            args.batch_size,
            args.max_tokens,
            save_callback,
            inference=args.inference,
            openai_client=openai_client,
            api_model_name=resolved_model_name,
            concurrent_requests=concurrent_requests,
            openai_timeout=openai_timeout,
        )
        all_results[dataset_name] = results
    
    print(f"\nAll results saved incrementally during processing")
    
    # Print summary
    print("\n=== Experiment Summary ===")
    for dataset_name, dataset_results in all_results.items():
        print(f"\n{dataset_name}:")
        for condition, condition_results in dataset_results.items():
            accuracy = condition_results["accuracy"]
            correct = condition_results["correct_count"]
            total = condition_results["total_count"]
            print(f"  {condition}: {accuracy:.3f} ({correct}/{total})")

if __name__ == "__main__":
    main()
