#!/usr/bin/env python3

"""
exp2_ablation.py - Multi-Agent Debate Ablation (No Misinformation) using MALLM

Runs the same multi-agent configuration as exp2.py, but with NO misinformation:
- All agents receive no extra "informations" (i.e., [None, None, None]).
- Only a single condition is run: "no_misinformation".

Outputs:
  out/<model_name>/exp2_ablation_results_<dataset>.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid

from typing import Optional

# Reuse endpoint discovery/auto-start and MALLM-path helpers from exp2.py
import exp2 as exp2_base


def _safe_model_name(model_name: str) -> str:
    safe = (model_name or "").strip().replace("/", "_")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", safe)
    return safe or "unknown_model"


def _read_json(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def run_with_mallm_no_misinformation(
    mock: bool,
    endpoint_url: str,
    model_name: str,
    debug_mode: bool = False,
    dataset_filter: str = "all",
) -> None:
    """
    Run a 3-agent debate using MALLM across datasets, with no misinformation at all.
    """
    exp2_base._ensure_mallm_on_path()
    if mock:
        exp2_base._install_runtime_stubs()
        # MALLM imports APIConnectionError from openai; exp2's lightweight stub may
        # omit it depending on version. Ensure it's present for mock runs.
        try:
            import openai  # type: ignore

            if not hasattr(openai, "APIConnectionError"):
                class APIConnectionError(Exception):  # noqa: N801
                    pass

                openai.APIConnectionError = APIConnectionError  # type: ignore[attr-defined]
        except Exception:
            pass

    from mallm import scheduler  # type: ignore
    from mallm.utils.config import Config  # type: ignore

    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(os.path.dirname(repo_root), "misinformed_agents", "data")

    os.makedirs("/tmp", exist_ok=True)
    os.makedirs("out", exist_ok=True)
    model_name_suffix = "mock_model" if mock else _safe_model_name(model_name)
    mallm_io_dir = os.path.join("/tmp", f"exp2_ablation_mallm_io_{model_name_suffix}")
    out_dir = os.path.join("out", model_name_suffix)
    os.makedirs(mallm_io_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    datasets_to_run = [
        {
            "name": "winogrande",
            "src": os.path.join(data_dir, "winogrande_misinformed.json"),
            "instruction_template": "winogrande",
            "instruction": None,
        },
        {
            "name": "ethics",
            "src": os.path.join(data_dir, "ethics_commonsense_misinformed.json"),
            "instruction_template": None,
            "instruction": "Would most people believe this is A) Ethical or B) Unethical? Answer with the letter only and include 'Final Solution: <letter>'.",
        },
        {
            "name": "cwq",
            "src": os.path.join(data_dir, "complex_web_questions_misinformed.json"),
            "instruction_template": None,
            "instruction": "Provide a clear and accurate answer to the question. Include 'Final Solution: <answer>' at the end.",
        },
    ]

    if isinstance(dataset_filter, str) and dataset_filter.lower() in ("winogrande", "ethics", "cwq"):
        datasets_to_run = [d for d in datasets_to_run if d["name"] == dataset_filter.lower()]

    def _convert_baseline(spec: dict, dst_path: str) -> None:
        """
        Convert <dataset>_misinformed.json to MALLM InputExample list, but with NO misinformation.
        """
        src = spec["src"]
        name = spec["name"]
        data = _read_json(src)
        if debug_mode and isinstance(data, list):
            data = data[:5]

        out: list[dict] = []
        for it in data:
            if name == "winogrande":
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
                references = [correct] if correct else []
            elif name == "ethics":
                sentence = it.get("sentence", it.get("input", ""))
                options = it.get("options", ["Ethical", "Unethical"])
                answer = it.get("answer", it.get("label", 0))
                try:
                    idx = int(answer)
                except Exception:
                    idx = 0
                correct = ""
                if isinstance(options, list) and 0 <= idx < len(options):
                    correct = str(options[idx])
                input_str = sentence
                if isinstance(options, list) and len(options) >= 2:
                    input_str += "\nOptions:\nA) " + str(options[0]) + "\nB) " + str(options[1])
                references = [correct] if correct else []
            else:
                question = it.get("question") or it.get("sentence") or it.get("webqsp_question") or ""
                input_str = str(question)
                references: list[str] = []
                ans = it.get("answers") or {}
                if isinstance(ans, dict) and isinstance(ans.get("answer"), list):
                    references = [str(x) for x in ans["answer"] if isinstance(x, str)]

            out.append(
                {
                    "example_id": it.get("id", it.get("ID", str(uuid.uuid4()))),
                    "dataset_id": name,
                    "inputs": [input_str],
                    "context": None,
                    "references": references,
                    "metadata": None,
                    # Key ablation: no misinformation for any agent.
                    "informations": [None, None, None],
                }
            )

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with open(dst_path, "w") as f:
            json.dump(out, f)

    def _evaluate_and_aggregate(dataset_name: str, condition: str, output_path: str) -> dict:
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

            is_mc = dataset_name in ("winogrande", "ethics")
            is_correct = False
            if is_mc:
                if dataset_name == "winogrande":
                    if isinstance(refs, list) and refs:
                        corr_text = str(refs[0]).lower()
                        is_correct = corr_text in predicted.lower()
                else:
                    if isinstance(refs, list) and refs:
                        corr_text = str(refs[0]).lower()
                        is_correct = (
                            ("a" in predicted.lower() and corr_text.startswith("ethical"))
                            or ("b" in predicted.lower() and corr_text.startswith("unethical"))
                            or (corr_text in predicted.lower())
                        )
            else:
                if isinstance(refs, list) and refs:
                    for r in refs:
                        if isinstance(r, str) and r and r.lower() in predicted.lower():
                            is_correct = True
                            break

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
                    "is_multiple_choice": is_mc,
                    "options": None,
                    "misinformation_strategy": condition,
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

    def _save_aggregate(dataset_name: str, condition: str, condition_results: dict) -> None:
        out_path = os.path.join(out_dir, f"exp2_ablation_results_{dataset_name}.json")
        existing: dict = {}
        if os.path.exists(out_path):
            try:
                with open(out_path, "r") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing[condition] = condition_results
        with open(out_path, "w") as f:
            json.dump(existing, f)

    condition = "no_misinformation"
    for spec in datasets_to_run:
        if not os.path.exists(spec["src"]):
            print(f"Warning: dataset not found, skipping: {spec['src']}")
            continue

        input_path = os.path.join(mallm_io_dir, f"exp2_ablation_mallm_input_{spec['name']}.json")
        output_path = os.path.join(mallm_io_dir, f"exp2_ablation_mallm_output_{spec['name']}.json")
        _convert_baseline(spec, input_path)

        if spec["instruction_template"]:
            cfg = Config(
                input_json_file_path=input_path,
                output_json_file_path=output_path,
                task_instruction_prompt="",
                task_instruction_prompt_template=spec["instruction_template"],
                endpoint_url=endpoint_url,
                model_name=model_name,
                discussion_paradigm="memory",
                decision_protocol="simple_voting",
                max_turns=5,
                num_agents=3,
                agent_generator="informed",
                agent_generators_list=["informed", "informed", "informed"],
                use_chain_of_thought=False,
                concurrent_api_requests=250,
                shuffle_input_samples=False,
            )
        else:
            cfg = Config(
                input_json_file_path=input_path,
                output_json_file_path=output_path,
                task_instruction_prompt=spec["instruction"] or "Answer the question. Provide the final answer clearly.",
                endpoint_url=endpoint_url,
                model_name=model_name,
                discussion_paradigm="memory",
                decision_protocol="simple_voting",
                max_turns=5,
                num_agents=3,
                agent_generator="informed",
                agent_generators_list=["informed", "informed", "informed"],
                use_chain_of_thought=False,
                concurrent_api_requests=250,
                shuffle_input_samples=False,
            )

        mallm_scheduler = scheduler.Scheduler(cfg)
        mallm_scheduler.run()

        cond_results = _evaluate_and_aggregate(spec["name"], condition, output_path)
        _save_aggregate(spec["name"], condition, cond_results)
        agg_path = os.path.join(out_dir, f"exp2_ablation_results_{spec['name']}.json")
        print(f"MALLM ablation complete for {spec['name']} [{condition}]. Output: {output_path}")
        print(f"Saved aggregated results to: {agg_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-agent debate (MALLM) across datasets with NO misinformation (ablation)."
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
    parser.add_argument("--debug", action="store_true", help="Debug mode: process only 5 samples per dataset.")
    parser.add_argument("--dataset", type=str, choices=["all", "winogrande", "ethics", "cwq"], default="all", help="Choose a single dataset to run or 'all'.")
    args = parser.parse_args()

    if args.mock:
        run_with_mallm_no_misinformation(True, "mock://local", "gpt-3.5-turbo", debug_mode=args.debug, dataset_filter=args.dataset)
        return

    resolved_endpoint = exp2_base._env_or_default_endpoint(args.endpoint_url)
    if resolved_endpoint.startswith("http") and not exp2_base._endpoint_alive(resolved_endpoint, timeout=1.5):
        auto = exp2_base._discover_endpoint_from_mallm(args.model_name)
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
                    started = exp2_base._start_model_server(args.model_name, desired_port=desired_port)
                    if started:
                        resolved_endpoint = started["endpoint_url"]  # type: ignore[index]
                    else:
                        print("[ERROR] Could not auto-start a model server. Please start one manually or provide a reachable --endpoint_url.")
                except Exception as e:
                    print(f"[ERROR] Auto-start failed: {e}. Please start a model server or set --no_autostart.")
            else:
                print("[INFO] Auto-start disabled (--no_autostart). Proceeding without starting a local server.")

    run_with_mallm_no_misinformation(False, resolved_endpoint, args.model_name, debug_mode=args.debug, dataset_filter=args.dataset)


if __name__ == "__main__":
    main()

