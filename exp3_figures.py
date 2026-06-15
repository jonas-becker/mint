#!/usr/bin/env python3

"""
exp3_figures.py - Visualization for 5-Agent Debate Experiment Results
This script creates visualizations for exp3 results comparing different agent configurations.
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import pandas as pd
import numpy as np
import os
import argparse
import re
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

# Import shared utilities and visualization functions
from shared_utils import load_results, find_result_files, infer_model_from_result_path
from shared_visualization import (
    setup_plot_style, save_plot, create_bar_plot, create_line_plot,
    create_heatmap, create_pie_chart, add_value_labels, get_named_colors
)

# Set up plot styling
setup_plot_style()

_DATASET_LABEL = {
    "complex_web_questions": "Complex Web Questions",
    "ethics": "Ethics",
    "logiqa": "LogiQA",
    "winogrande": "WinoGrande",
    "cwq": "Complex Web Questions",
}


def _display_dataset_name(ds: str) -> str:
    ds = str(ds)
    if ds in _DATASET_LABEL:
        return _DATASET_LABEL[ds]
    return ds.replace("_", " ").strip().title()

def _parse_condition_counts(condition: str) -> tuple[Optional[int], Optional[int]]:
    """Return (misinformed_count, informed_count) if parseable from condition key."""
    m = re.match(r"^\s*(\d+)\s*_misinformed_(\d+)\s*_informed\s*$", str(condition))
    if not m:
        return None, None
    try:
        return int(m.group(1)), int(m.group(2))
    except Exception:
        return None, None


def _display_condition_name(condition: str) -> str:
    """Human-friendly condition label for plots."""
    mis, inf = _parse_condition_counts(condition)
    if mis is not None and inf is not None:
        mis_s = "misinformed" if mis == 1 else "misinformed"
        inf_s = "uninformed" if inf == 1 else "uninformed"
        # Newline keeps labels compact without rotation.
        return f"{mis} {mis_s}\n{inf} {inf_s}"
    return str(condition).replace("_", " ").strip().title()


def _display_model_name_short(model_name: str) -> str:
    """Short legend label for known exp3 models."""
    raw = str(model_name).lower()
    if "llama" in raw and "3.3" in raw:
        return "Llama-3.3"
    if "glm" in raw and "4.7" in raw:
        return "GLM-4.7"
    if "gpt-oss" in raw:
        return "GPT-OSS-120B"
    disp = str(model_name).replace("_", "/", 1)
    if "/" in disp:
        disp = disp.split("/", 1)[1]
    return disp


def _model_order_key(m: str) -> tuple[int, str]:
    s = str(m).lower()
    if "llama" in s:
        return (0, s)
    if "glm" in s:
        return (1, s)
    if "gpt-oss" in s:
        return (2, s)
    return (3, s)


def _condition_sort_key(condition: str) -> tuple[int, int, str]:
    """Sort configurations by misinformed count, then informed count."""
    mis, inf = _parse_condition_counts(condition)
    if mis is None or inf is None:
        return (10_000, 10_000, str(condition))
    return (mis, inf, str(condition))


def _tilt_xlabels(ax: Axes, rotation: int = 35) -> None:
    """Tilt and right-align x tick labels to avoid overlaps."""
    ax.tick_params(axis="x", labelrotation=rotation)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
        lab.set_rotation_mode("anchor")

def _canonical_dataset_name(name: str) -> str:
    # Keep filenames stable but make plots readable.
    mapping = {
        "cwq": "complex_web_questions",
    }
    return mapping.get(name, name)


def _canon_answer(x: Optional[object]) -> Optional[str]:
    """Canonicalize an answer/solution string for robust equality checks."""
    if x is None:
        return None
    try:
        s = str(x).strip()
    except Exception:
        return None
    if not s:
        return None

    # Strip common wrappers
    s = re.sub(r"^\s*(final\s+(answer|solution)\s*:)\s*", "", s, flags=re.IGNORECASE)
    s = s.strip()

    # Remove leading multiple-choice option markers: "A) ", "(b).", "c:" ...
    s = re.sub(r"^\s*[\(\[]?\s*[a-e]\s*[\)\].:\-]\s*", "", s, flags=re.IGNORECASE)

    # Collapse whitespace, strip punctuation at ends, normalize case
    s = " ".join(s.split()).strip().strip(" .,:;\"'`")
    s = s.lower()
    return s or None


def _answer_matches(solution: Optional[str], correct: Optional[str]) -> bool:
    """Heuristic match between agent solution and ground-truth answer."""
    a = _canon_answer(solution)
    b = _canon_answer(correct)
    if a is None or b is None:
        return False
    if a == b:
        return True
    # Some agents include extra wording; allow substring match for non-trivial answers.
    if len(b) >= 3 and b in a:
        return True
    if len(a) >= 3 and a in b:
        return True
    return False


def _is_agreement_only_solution(solution: str) -> bool:
    """Heuristic: solution indicates agreement/unchanged rather than an actual answer."""
    s = " ".join(str(solution).strip().lower().split())
    if not s:
        return True

    agreement_starts = (
        "i agree",
        "agree",
        "same",
        "same as",
        "as above",
        "unchanged",
        "no change",
        "no changes",
        "nothing to add",
    )
    if any(s.startswith(p) for p in agreement_starts):
        return True

    # Very short acknowledgements
    if s in {"yes", "yep", "yeah", "correct", "true", "ok", "okay"}:
        return True

    return False


def _is_misinformed_persona_description(desc: Optional[str]) -> bool:
    if desc is None:
        return False
    s = " ".join(str(desc).strip().lower().split())
    if not s:
        return False
    # In the stored exp3 MALLM logs, misinformed agents get an extra snippet:
    # "A participant of the discussion with the following information: ..."
    return "following information:" in s


def _agent_persona_descriptions(mallm_log: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Extract agent_id -> persona_description from globalMemory additional_args."""
    if not isinstance(mallm_log, dict):
        return {}
    mem = mallm_log.get("globalMemory")
    if not isinstance(mem, list) or not mem:
        return {}

    out: Dict[str, str] = {}
    for m in mem:
        if not isinstance(m, dict):
            continue
        aid = m.get("agent_id") or m.get("agentId")
        if not isinstance(aid, str) or not aid.strip():
            continue
        aa = m.get("additional_args")
        if not isinstance(aa, dict):
            continue
        pd = aa.get("persona_description")
        if pd is None:
            continue
        try:
            pd_s = str(pd).strip()
        except Exception:
            continue
        if not pd_s:
            continue
        out.setdefault(aid, pd_s)
    return out

def _extract_solutions_by_agent_turn(mallm_log: Optional[Dict[str, Any]]) -> Dict[str, Dict[int, str]]:
    """Extract per-agent per-turn latest solution from mallm_log.globalMemory."""
    if not isinstance(mallm_log, dict):
        return {}
    mem = mallm_log.get("globalMemory")
    if not isinstance(mem, list) or not mem:
        return {}

    latest: Dict[tuple[str, int], tuple[int, Optional[str]]] = {}
    for m in mem:
        if not isinstance(m, dict):
            continue
        aid = m.get("agent_id") or m.get("agentId")
        if not isinstance(aid, str) or not aid.strip():
            continue
        try:
            turn = int(m.get("turn"))
        except Exception:
            continue
        mid = m.get("message_id")
        try:
            mid_i = int(mid) if mid is not None else -1
        except Exception:
            mid_i = -1

        sol = m.get("solution")
        if sol is None:
            continue
        try:
            sol_s = str(sol).strip()
        except Exception:
            continue
        if not sol_s:
            continue

        key = (aid, turn)
        prev = latest.get(key)
        if prev is None or mid_i >= prev[0]:
            latest[key] = (mid_i, sol_s)

    out: Dict[str, Dict[int, str]] = {}
    for (aid, turn), (_mid, sol) in latest.items():
        if sol is None:
            continue
        out.setdefault(aid, {})[turn] = sol
    return out


def _persona_name_map(mallm_log: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not isinstance(mallm_log, dict):
        return {}
    personas = mallm_log.get("personas")
    if not isinstance(personas, list):
        return {}
    out: Dict[str, str] = {}
    for p in personas:
        if not isinstance(p, dict):
            continue
        aid = p.get("agentId") or p.get("agent_id")
        if not isinstance(aid, str) or not aid.strip():
            continue
        name = p.get("persona") or p.get("personaName") or p.get("agentName") or aid
        out[aid] = str(name)
    return out


def extract_agent_solutions_data(
    results: Any, dataset_name: Optional[str] = None, model_name: Optional[str] = None
) -> List[Dict]:
    """Extract agent solutions data from exp3 results.

    Supports the repo's current exp3 schema:
      out/exp3_results_{dataset}.json = {condition_key: {accuracy, ..., results: [...]}, ...}
    Each result item contains mallm_log.globalMemory with per-agent per-turn solutions.
    """
    data: List[Dict] = []

    # Back-compat: if someone passes a list of rows that already includes agent_solutions.
    if isinstance(results, list):
        for result in results:
            if not isinstance(result, dict):
                continue
            if "agent_solutions" in result and isinstance(result["agent_solutions"], dict):
                for agent_id, solutions in result["agent_solutions"].items():
                    if not isinstance(solutions, list):
                        continue
                    for solution in solutions:
                        if not isinstance(solution, dict):
                            continue
                        data.append(
                            {
                                "dataset": result.get("dataset_name", dataset_name or "unknown"),
                                "model": str(model_name or "combined"),
                                "condition": result.get("experimental_condition", "unknown"),
                                "turn": solution.get("turn"),
                                "agent": solution.get("agent"),
                                "agent_id": agent_id,
                                "agent_is_correct": bool(solution.get("is_correct", False)),
                                "misinformation_strategy": result.get("misinformation_strategy", "unknown"),
                                "is_multiple_choice": bool(result.get("is_multiple_choice", True)),
                                "majority_is_correct": bool(result.get("is_correct", False)),
                            }
                        )
        return data

    # Current schema: dict of condition -> payload
    if not isinstance(results, dict):
        return data

    for condition_key, payload in results.items():
        if not isinstance(payload, dict):
            continue
        rows = payload.get("results")
        if not isinstance(rows, list) or not rows:
            continue

        for r in rows:
            if not isinstance(r, dict):
                continue
            ml = r.get("mallm_log")
            ds = dataset_name
            if ds is None and isinstance(ml, dict):
                ds = ml.get("dataset") or ml.get("datasetId")
            ds = _canonical_dataset_name(str(ds)) if ds is not None else "unknown"

            correct_answer = r.get("correct_answer")
            persona_map = _persona_name_map(ml)
            by_agent_turn = _extract_solutions_by_agent_turn(ml)
            if not by_agent_turn:
                continue

            for aid, by_turn in by_agent_turn.items():
                agent_name = persona_map.get(aid, aid)
                for turn, sol in by_turn.items():
                    data.append(
                        {
                            "dataset": ds,
                            "model": str(model_name or "combined"),
                            "condition": str(condition_key),
                            "turn": int(turn),
                            "agent": str(agent_name),
                            "agent_id": str(aid),
                            "agent_is_correct": _answer_matches(sol, correct_answer),
                            "misinformation_strategy": r.get("misinformation_strategy", "unknown"),
                            "is_multiple_choice": bool(r.get("is_multiple_choice", True)),
                            "majority_is_correct": bool(r.get("is_correct", False)),
                        }
                    )

    return data


def extract_last_agent_data(
    results: Any, dataset_name: Optional[str] = None, model_name: Optional[str] = None
) -> List[Dict]:
    """Extract *last speaking agent per turn* accuracy (ignores majority vote).

    For each debate instance (result row), and for each debate turn, we pick the last
    `mallm_log.globalMemory` entry *within that turn* (the last speaking agent).

    If that last agent's `solution` is empty or just agreement/unchanged, we score the
    previous substantive `solution` from the same turn as the last agent's endorsed answer.

    The resulting accuracy averages (in plots) are computed only over these last-per-turn solutions.
    """
    out: List[Dict] = []

    if not isinstance(results, dict):
        return out

    for condition_key, payload in results.items():
        if not isinstance(payload, dict):
            continue
        rows = payload.get("results")
        if not isinstance(rows, list) or not rows:
            continue

        for r in rows:
            if not isinstance(r, dict):
                continue
            ml = r.get("mallm_log")
            if not isinstance(ml, dict):
                continue

            ds = dataset_name
            if ds is None:
                ds = ml.get("dataset") or ml.get("datasetId")
            ds = _canonical_dataset_name(str(ds)) if ds is not None else "unknown"

            correct_answer = r.get("correct_answer")
            persona_map = _persona_name_map(ml)

            mem = ml.get("globalMemory")
            if not isinstance(mem, list) or not mem:
                continue

            # Group messages by turn (preserve order) so we can resolve "agreement-only" last answers.
            msgs_by_turn: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            for m in mem:
                if not isinstance(m, dict):
                    continue
                try:
                    turn = int(m.get("turn", -1))
                except Exception:
                    continue
                if turn < 0:
                    continue
                msgs_by_turn[turn].append(m)

            if not msgs_by_turn:
                continue

            for turn in sorted(msgs_by_turn.keys()):
                msgs = msgs_by_turn[turn]
                last_msg = msgs[-1]

                # The last agent is whoever produced the last message of the turn.
                aid = last_msg.get("agent_id") or last_msg.get("agentId") or "unknown"
                agent_name = persona_map.get(str(aid), str(aid))

                # If last agent's "solution" is just agreement/unchanged, use the previous substantive
                # solution within the same turn (most recent non-empty, non-agreement `solution`).
                effective_solution = last_msg.get("solution")
                effective_solution_s = ""
                if effective_solution is not None:
                    try:
                        effective_solution_s = str(effective_solution).strip()
                    except Exception:
                        effective_solution_s = ""

                if effective_solution is None or not effective_solution_s or _is_agreement_only_solution(effective_solution_s):
                    fallback = None
                    for prev in reversed(msgs[:-1]):
                        sol = prev.get("solution")
                        if sol is None:
                            continue
                        try:
                            sol_s = str(sol).strip()
                        except Exception:
                            continue
                        if not sol_s or _is_agreement_only_solution(sol_s):
                            continue
                        fallback = sol
                        break
                    if fallback is not None:
                        effective_solution = fallback

                out.append(
                    {
                        "dataset": ds,
                        "model": str(model_name or "combined"),
                        "condition": str(condition_key),
                        "turn": int(turn),
                        "last_turn": int(turn),  # back-compat name used elsewhere
                        "last_agent": str(agent_name),
                        "last_agent_id": str(aid),
                        "last_agent_is_correct": _answer_matches(effective_solution, correct_answer),
                        "misinformation_strategy": r.get("misinformation_strategy", "unknown"),
                        "is_multiple_choice": bool(r.get("is_multiple_choice", True)),
                        "majority_is_correct": bool(r.get("is_correct", False)),
                    }
                )

    return out


def extract_vote_data(
    results: Any, dataset_name: Optional[str] = None, model_name: Optional[str] = None
) -> List[Dict]:
    """Extract per-agent vote correctness from exp3 results.

    Uses `mallm_log.votesEachTurn` (final voting turn) and evaluates each voter's selected
    answer against `correct_answer`.
    """
    out: List[Dict] = []
    if not isinstance(results, dict):
        return out

    for condition_key, payload in results.items():
        if not isinstance(payload, dict):
            continue
        rows = payload.get("results")
        if not isinstance(rows, list) or not rows:
            continue

        for r in rows:
            if not isinstance(r, dict):
                continue
            ml = r.get("mallm_log")
            if not isinstance(ml, dict):
                continue

            ds = dataset_name
            if ds is None:
                ds = ml.get("dataset") or ml.get("datasetId")
            ds = _canonical_dataset_name(str(ds)) if ds is not None else "unknown"

            correct_answer = r.get("correct_answer")
            personas = ml.get("personas")
            if not isinstance(personas, list) or not personas:
                continue

            votes_each_turn = ml.get("votesEachTurn")
            if not isinstance(votes_each_turn, dict) or not votes_each_turn:
                continue

            # Use the last (highest) turn key available.
            turn_keys: List[int] = []
            for k in votes_each_turn.keys():
                try:
                    turn_keys.append(int(k))
                except Exception:
                    continue
            if not turn_keys:
                continue
            last_turn = max(turn_keys)
            entry = votes_each_turn.get(str(last_turn)) or votes_each_turn.get(last_turn)
            if not isinstance(entry, dict):
                continue

            answers = entry.get("answers")
            if not isinstance(answers, list) or not answers:
                continue

            alterations = entry.get("alterations")
            if not isinstance(alterations, dict) or not alterations:
                continue
            alt_key = "anonymous" if "anonymous" in alterations else next(iter(alterations.keys()))
            alt = alterations.get(alt_key)
            if not isinstance(alt, dict):
                continue
            votes = alt.get("votes")
            if not isinstance(votes, list) or not votes:
                continue

            # votes[i] = index of the *answer* the i-th agent voted for.
            for voter_idx, chosen_idx in enumerate(votes):
                if voter_idx >= len(personas):
                    continue
                p = personas[voter_idx]
                if not isinstance(p, dict):
                    continue
                aid = p.get("agentId") or p.get("agent_id") or str(voter_idx)
                aname = p.get("persona") or p.get("personaName") or p.get("agentName") or f"Agent {voter_idx+1}"

                try:
                    ci = int(chosen_idx)
                except Exception:
                    continue
                if not (0 <= ci < len(answers)):
                    continue

                selected_answer = answers[ci]
                voted_correct = _answer_matches(selected_answer, correct_answer)

                out.append(
                    {
                        "dataset": ds,
                        "model": str(model_name or "combined"),
                        "condition": str(condition_key),
                        "turn": int(last_turn),
                        "alteration": str(alt_key),
                        "voter": str(aname),
                        "voter_id": str(aid),
                        "voted_for_answer_idx": int(ci),
                        "voted_answer": str(selected_answer),
                        "voted_is_correct": bool(voted_correct),
                    }
                )

    return out


def extract_last_agent_vote_data(
    results: Any, dataset_name: Optional[str] = None, model_name: Optional[str] = None
) -> List[Dict]:
    """Extract *last speaking agent per turn* vote correctness.

    For each debate instance and each available voting turn in `mallm_log.votesEachTurn`, we:
    - identify the last speaking agent in that same `turn` from `mallm_log.globalMemory`
    - take that agent's vote from `votesEachTurn[turn].alterations[*].votes`
    - evaluate the voted-for answer against `correct_answer`

    Note: Some logs only contain a single voting turn (e.g., final turn). In that case, this will
    yield one record per debate instance.
    """
    out: List[Dict] = []
    if not isinstance(results, dict):
        return out

    for condition_key, payload in results.items():
        if not isinstance(payload, dict):
            continue
        rows = payload.get("results")
        if not isinstance(rows, list) or not rows:
            continue

        for r in rows:
            if not isinstance(r, dict):
                continue
            ml = r.get("mallm_log")
            if not isinstance(ml, dict):
                continue

            ds = dataset_name
            if ds is None:
                ds = ml.get("dataset") or ml.get("datasetId")
            ds = _canonical_dataset_name(str(ds)) if ds is not None else "unknown"

            correct_answer = r.get("correct_answer")
            persona_map = _persona_name_map(ml)

            mem = ml.get("globalMemory")
            if not isinstance(mem, list) or not mem:
                continue

            # Build turn -> messages mapping (preserve order)
            msgs_by_turn: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            for m in mem:
                if not isinstance(m, dict):
                    continue
                try:
                    turn = int(m.get("turn", -1))
                except Exception:
                    continue
                if turn < 0:
                    continue
                msgs_by_turn[turn].append(m)

            personas = ml.get("personas")
            if not isinstance(personas, list) or not personas:
                continue
            agent_id_to_idx: Dict[str, int] = {}
            for idx, p in enumerate(personas):
                if not isinstance(p, dict):
                    continue
                aid = p.get("agentId") or p.get("agent_id")
                if aid is None:
                    continue
                agent_id_to_idx[str(aid)] = idx

            votes_each_turn = ml.get("votesEachTurn")
            if not isinstance(votes_each_turn, dict) or not votes_each_turn:
                continue

            for turn_key, entry in votes_each_turn.items():
                try:
                    turn = int(turn_key)
                except Exception:
                    continue
                if not isinstance(entry, dict):
                    continue

                msgs = msgs_by_turn.get(turn)
                if not msgs:
                    continue
                last_msg = msgs[-1]

                last_aid = last_msg.get("agent_id") or last_msg.get("agentId")
                last_aid_s = str(last_aid) if last_aid is not None else "unknown"
                voter_idx = agent_id_to_idx.get(last_aid_s)
                if voter_idx is None:
                    continue

                answers = entry.get("answers")
                if not isinstance(answers, list) or not answers:
                    continue

                alterations = entry.get("alterations")
                if not isinstance(alterations, dict) or not alterations:
                    continue
                alt_key = "anonymous" if "anonymous" in alterations else next(iter(alterations.keys()))
                alt = alterations.get(alt_key)
                if not isinstance(alt, dict):
                    continue
                votes = alt.get("votes")
                if not isinstance(votes, list) or not votes:
                    continue
                if voter_idx >= len(votes):
                    continue

                try:
                    chosen_idx = int(votes[voter_idx])
                except Exception:
                    continue
                if not (0 <= chosen_idx < len(answers)):
                    continue

                selected_answer = answers[chosen_idx]
                voted_correct = _answer_matches(selected_answer, correct_answer)

                out.append(
                    {
                        "dataset": ds,
                        "model": str(model_name or "combined"),
                        "condition": str(condition_key),
                        "turn": int(turn),
                        "alteration": str(alt_key),
                        "last_voter": str(persona_map.get(last_aid_s, last_aid_s)),
                        "last_voter_id": last_aid_s,
                        "last_voter_voted_for_answer_idx": int(chosen_idx),
                        "last_voter_voted_answer": str(selected_answer),
                        "last_voter_voted_is_correct": bool(voted_correct),
                    }
                )

    return out


def extract_misinformed_adjustment_data(
    results: Any, dataset_name: Optional[str] = None, model_name: Optional[str] = None
) -> List[Dict]:
    """Measure whether misinformed agents adjust toward the correct answer.

    For each debate instance and each misinformed agent:
    - determine correctness at turn 1 (initial)
    - determine correctness at the agent's last available turn
    - define "adjusted_to_correct" as started_wrong AND ended_correct

    We later aggregate this as P(adjusted_to_correct | started_wrong) by informed_count.
    """
    out: List[Dict] = []
    if not isinstance(results, dict):
        return out

    for condition_key, payload in results.items():
        if not isinstance(payload, dict):
            continue
        rows = payload.get("results")
        if not isinstance(rows, list) or not rows:
            continue

        for r in rows:
            if not isinstance(r, dict):
                continue
            ml = r.get("mallm_log")
            if not isinstance(ml, dict):
                continue

            ds = dataset_name
            if ds is None:
                ds = ml.get("dataset") or ml.get("datasetId")
            ds = _canonical_dataset_name(str(ds)) if ds is not None else "unknown"

            correct_answer = r.get("correct_answer")
            meta = ml.get("metadata") if isinstance(ml.get("metadata"), dict) else {}
            informed_count = meta.get("informed_count")
            misinformed_count = meta.get("misinformed_count")
            try:
                informed_count_i = int(informed_count) if informed_count is not None else None
            except Exception:
                informed_count_i = None
            try:
                misinformed_count_i = int(misinformed_count) if misinformed_count is not None else None
            except Exception:
                misinformed_count_i = None

            by_agent_turn = _extract_solutions_by_agent_turn(ml)
            if not by_agent_turn:
                continue

            persona_desc = _agent_persona_descriptions(ml)
            if not persona_desc:
                continue

            # Identify misinformed agents via persona_description marker.
            mis_agents = [aid for aid, pd in persona_desc.items() if _is_misinformed_persona_description(pd)]
            if not mis_agents:
                continue

            for aid in mis_agents:
                turns = by_agent_turn.get(aid)
                if not isinstance(turns, dict) or not turns:
                    continue
                # Turn 1 solution (initial)
                init_sol = turns.get(1)
                if init_sol is None:
                    continue
                # Last available turn for that agent
                last_turn = max(int(t) for t in turns.keys())
                last_sol = turns.get(last_turn)
                if last_sol is None:
                    continue

                init_correct = _answer_matches(init_sol, correct_answer)
                last_correct = _answer_matches(last_sol, correct_answer)
                started_wrong = not init_correct
                adjusted_to_correct = bool(started_wrong and last_correct)

                out.append(
                    {
                        "dataset": ds,
                        "model": str(model_name or "combined"),
                        "condition": str(condition_key),
                        "informed_count": informed_count_i,
                        "misinformed_count": misinformed_count_i,
                        "agent_id": str(aid),
                        "started_wrong": bool(started_wrong),
                        "adjusted_to_correct": bool(adjusted_to_correct),
                    }
                )

    return out

def create_agent_configuration_analysis(data: List[Dict], output_dir: str = "out/figures/exp3"):
    """Create analysis of different agent configurations."""
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No data available for agent configuration analysis")
        return
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    df["dataset_display"] = df["dataset"].map(_display_dataset_name)
    
    model_list = []
    if "model" in df.columns:
        model_list = sorted(df["model"].dropna().astype(str).unique().tolist())
    multi_model = len(model_list) > 1

    # Plot 1: Overall accuracy by condition
    ax1 = axes[0, 0]
    condition_accuracy = df.groupby(["condition"] + (["model"] if multi_model else []), observed=True)["majority_is_correct"].mean().reset_index()
    conds = sorted(condition_accuracy["condition"].astype(str).unique().tolist(), key=_condition_sort_key)
    vals = [
        float(
            condition_accuracy[condition_accuracy["condition"].astype(str) == c]["majority_is_correct"].mean()
        )
        for c in conds
    ]
    labels = [_display_condition_name(c) for c in conds]
    mis_labels = []
    for c in conds:
        mis, _inf = _parse_condition_counts(str(c))
        mis_labels.append(str(mis) if mis is not None else str(c))
    # Ordered gradient left→right: green → red.
    from matplotlib.colors import LinearSegmentedColormap
    from figure_style import sample_cmap
    cmap = LinearSegmentedColormap.from_list("green_to_red_overall_acc", ["#2e7d32", "#c62828"])
    colors = sample_cmap(cmap, len(conds))
    x = np.arange(len(conds))
    if multi_model:
        from shared_visualization import get_palette
        pal = get_palette()
        model_display = [m.replace("_", "/", 1) for m in model_list]
        color_map = {m: pal[i % len(pal)] for i, m in enumerate(model_display)}
        condition_accuracy["condition"] = condition_accuracy["condition"].astype(str)
        condition_accuracy["condition_display"] = condition_accuracy["condition"].map(_display_condition_name)
        condition_accuracy["model_display"] = condition_accuracy["model"].astype(str).str.replace("_", "/", n=1, regex=False)
        sns.barplot(
            data=condition_accuracy,
            x="condition_display",
            y="majority_is_correct",
            hue="model_display",
            order=labels,
            hue_order=model_display,
            palette=color_map,
            ax=ax1,
        )
        bars = []
        handles, legend_labels = ax1.get_legend_handles_labels()
        if getattr(ax1, "legend_", None) is not None:
            ax1.legend_.remove()
        fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=max(1, len(legend_labels)), frameon=True)
    else:
        bars = ax1.bar(x, vals, alpha=0.85, color=colors)
    x_tick_fs = 14
    y_tick_fs = 12
    value_fs = 12
    ax1.set_title("Overall Accuracy by Agent Configuration", fontweight="bold")
    ax1.set_xlabel("Agent Configuration")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=x_tick_fs)
    ax1.tick_params(axis="y", labelsize=y_tick_fs)
    # Add value labels
    if not multi_model:
        for bar, value in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2., value + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=value_fs, fontweight='bold')
    
    # Plot 2: Individual agent performance
    ax2 = axes[0, 1]
    agent_accuracy = df.groupby('agent')['agent_is_correct'].mean()
    bars = ax2.bar(agent_accuracy.index, agent_accuracy.values, alpha=0.7)
    ax2.set_title('Individual Agent Performance', fontweight='bold')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    _tilt_xlabels(ax2, rotation=45)
    # Add value labels
    for bar, value in zip(bars, agent_accuracy.values):
        ax2.text(bar.get_x() + bar.get_width()/2., value + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Accuracy by dataset and condition
    ax3 = axes[1, 0]
    dataset_condition_accuracy = df.groupby(["dataset_display", "condition"])["majority_is_correct"].mean().unstack()
    dataset_condition_accuracy = dataset_condition_accuracy.rename(columns={c: _display_condition_name(c) for c in dataset_condition_accuracy.columns})
    dataset_condition_accuracy.plot(kind='bar', ax=ax3, alpha=0.7)
    ax3.set_title('Accuracy by Dataset and Agent Configuration', fontweight='bold')
    ax3.set_ylabel('Accuracy')
    ax3.set_ylim(0, 1)
    ax3.legend(title='Configuration')
    _tilt_xlabels(ax3, rotation=45)
    
    # Plot 4: Turn evolution analysis
    ax4 = axes[1, 1]
    turn_accuracy = df.groupby(['turn', 'condition'])['agent_is_correct'].mean().unstack()
    for condition in turn_accuracy.columns:
        ax4.plot(turn_accuracy.index, turn_accuracy[condition], marker='o', label=_display_condition_name(str(condition)), linewidth=2)
    ax4.set_title('Agent Performance Evolution Across Turns', fontweight='bold')
    ax4.set_xlabel('Turn')
    ax4.set_ylabel('Average Agent Accuracy')
    ax4.set_ylim(0, 1)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(output_dir, "exp3_agent_configuration_analysis.pdf")

    # Standalone version of the top-left panel (paper-ready).
    fig2, ax = plt.subplots(figsize=(7.6 if not multi_model else 10.8, 6.0))
    slate = get_named_colors().get("slate", "#4A4A4A")
    base_fs = float(plt.rcParams.get("font.size", 12))
    x_tick_fs = max(10.0, base_fs * 0.68)
    y_tick_fs = max(12.0, base_fs * 0.80)
    value_fs = max(11.0, base_fs * 0.75)
    pct_fs = max(10.0, base_fs * 0.70)
    title_fs = max(14.0, base_fs * 1.00)
    legend_fs = max(10.0, base_fs * 0.70)
    ax.set_title("Overall Accuracy by Agent Configuration", fontweight="bold", fontsize=title_fs, pad=18)
    ax.set_xlabel("Agent Configuration", fontsize=x_tick_fs, labelpad=14)
    ax.set_ylabel("Accuracy", fontsize=y_tick_fs)
    ax.set_ylim(0, 1.06)
    # Match reference plot style: use condition-formatted x labels.
    bin_labels = [_display_condition_name(c) for c in conds]
    from figure_style import get_palette as get_fig_palette
    pal3 = get_fig_palette(3)
    if multi_model:
        from shared_visualization import get_palette
        pal = get_palette()
        model_pairs = []
        for m in model_list:
            disp_full = m.replace("_", "/", 1)
            disp = disp_full.split("/", 1)[1] if "/" in disp_full else disp_full
            model_pairs.append((m, disp))
        model_display = [disp for _raw, disp in model_pairs]
        if len(model_display) == 2:
            # Match the reference plot's purple/yellow pairing.
            model_colors = [pal3[0], pal3[2]]
        else:
            model_colors = [pal[i % len(pal)] for i in range(len(model_display))]
        color_map = {m: model_colors[i] for i, m in enumerate(model_display)}
        model_markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
        x_positions = np.arange(len(mis_labels))

        for i, (model_key, model_disp) in enumerate(model_pairs):
            y_vals = []
            for c in conds:
                rows = condition_accuracy[
                    (condition_accuracy["condition"].astype(str) == str(c))
                    & (condition_accuracy["model"].astype(str) == model_key)
                ]["majority_is_correct"]
                y_vals.append(float(rows.iloc[0]) if not rows.empty else np.nan)
            ax.plot(
                x_positions,
                y_vals,
                marker=model_markers[i % len(model_markers)],
                linewidth=2.8,
                markersize=7,
                color=color_map[model_disp],
                alpha=0.9,
                label=model_disp,
            )
            for xp, yv in zip(x_positions, y_vals):
                if np.isnan(yv):
                    continue
                ax.text(
                    xp,
                    min(1.04, float(yv) + 0.014),
                    f"{float(yv):.3f}".replace(".", ","),
                    ha="center",
                    va="bottom",
                    fontsize=value_fs - 1,
                    fontweight="normal",
                    color=slate,
                )

        handles, legend_labels = ax.get_legend_handles_labels()
        fig2.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=max(1, len(model_display)),
            frameon=True,
            title="",
            fontsize=legend_fs,
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(bin_labels, fontsize=x_tick_fs)
    else:
        ax.plot(x, vals, marker="o", linewidth=2.8, markersize=7.0, color=pal3[0], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=x_tick_fs)
    ax.tick_params(axis="y", labelsize=y_tick_fs)
    # Keep the green→red background tint behind x tick labels.
    tick_bg_colors = colors[: len(ax.get_xticklabels())]
    for lab, col in zip(ax.get_xticklabels(), tick_bg_colors):
        lab.set_bbox(
            dict(
                facecolor=col,
                edgecolor="none",
                boxstyle="round,pad=0.25",
                alpha=0.25,
            )
        )
    ax.margins(x=0.02)
    ax.grid(True, axis="y", alpha=0.25)
    first_val = float(vals[0]) if vals else float("nan")
    if not multi_model:
        for i, v in enumerate(vals):
            drop_pp = (first_val - float(v)) * 100.0 if np.isfinite(first_val) else np.nan
            xmid = x[i]
            if i == 0:
                y_abs = min(1.02, float(v) + 0.01)
            else:
                y_abs = min(1.04, float(v) + 0.05)
            y_pct = min(1.01, y_abs - 0.04)
            ax.text(
                xmid,
                y_abs,
                f"{float(v):.3f}".replace(".", ","),
                ha="center",
                va="bottom",
                fontsize=value_fs,
                fontweight="normal",
                color=slate,
            )
            if i == 0:
                continue
            ax.text(
                xmid,
                y_pct,
                f"(-{max(0.0, drop_pp):.1f}%)",
                ha="center",
                va="bottom",
                fontsize=pct_fs,
                fontweight="normal",
                color=slate,
            )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.88 if multi_model else 1.0))
    save_plot(output_dir, "exp3_overall_accuracy_by_agent_configuration.pdf")

def create_majority_vs_individual_analysis(data: List[Dict], output_dir: str = "out/figures/exp3"):
    """Create analysis comparing majority voting vs individual agent performance."""
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No data available for majority vs individual analysis")
        return
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(18, 13.5))
    
    # Plot 1: Majority vs Individual accuracy comparison
    ax1 = axes[0, 0]
    majority_accuracy = df.groupby("condition")["majority_is_correct"].mean()
    individual_accuracy = df.groupby("condition")["agent_is_correct"].mean()
    conds = sorted(majority_accuracy.index.astype(str).tolist(), key=_condition_sort_key)
    majority_accuracy = majority_accuracy.reindex(conds)
    individual_accuracy = individual_accuracy.reindex(conds)
    
    x = np.arange(len(majority_accuracy))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, majority_accuracy.values, width, label='Majority Vote', alpha=0.7)
    bars2 = ax1.bar(x + width/2, individual_accuracy.values, width, label='Individual Agents', alpha=0.7)
    
    ax1.set_title("Majority Vote vs Individual Agent Performance", fontweight="bold")
    ax1.set_ylabel('Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels([_display_condition_name(c) for c in conds])
    _tilt_xlabels(ax1, rotation=0)
    # Place a single legend for the whole figure (avoids overlapping subplot titles).
    handles, labels_ = ax1.get_legend_handles_labels()
    if getattr(ax1, "legend_", None) is not None:
        ax1.legend_.remove()
    ax1.set_ylim(0, 1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Performance improvement from majority voting
    ax2 = axes[0, 1]
    improvement = majority_accuracy - individual_accuracy
    try:
        import plot_config  # type: ignore

        _c = dict(getattr(plot_config, "MISINFO_RELEVANCE_COLORS", {}))
        _pos = _c.get("baseline", "#2E7D32")
        _neg = _c.get("relevant", "#C62828")
    except Exception:
        _pos = "#2E7D32"
        _neg = "#C62828"

    bars = ax2.bar(
        np.arange(len(improvement)),
        improvement.values,
        alpha=0.7,
        color=[_pos if v > 0 else _neg for v in improvement.values],
    )
    ax2.set_title("Performance Improvement from Majority Voting", fontweight="bold", pad=10)
    ax2.set_ylabel('Accuracy Improvement')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xticks(np.arange(len(improvement)))
    ax2.set_xticklabels([_display_condition_name(c) for c in conds])
    _tilt_xlabels(ax2, rotation=0)
    # Add value labels
    for bar, value in zip(bars, improvement.values):
        ax2.text(bar.get_x() + bar.get_width()/2., value + (0.01 if value > 0 else -0.01),
                f'{value:.3f}', ha='center', va='bottom' if value > 0 else 'top', fontweight='bold')
    
    # Plot 3: Agent agreement analysis
    ax3 = axes[1, 0]
    # Calculate agreement rate (how often agents agree with each other)
    agreement_data = []
    for condition in df["condition"].unique():
        condition_data = df[df["condition"] == condition]
        for turn in condition_data["turn"].unique():
            turn_data = condition_data[condition_data['turn'] == turn]
            if len(turn_data) >= 5:  # We have 5 agents
                # Calculate how many agents gave the same answer
                correct_answers = turn_data['agent_is_correct'].sum()
                agreement_rate = correct_answers / len(turn_data)
                agreement_data.append({
                    'condition': condition,
                    'turn': turn,
                    'agreement_rate': agreement_rate
                })
    
    if agreement_data:
        agreement_df = pd.DataFrame(agreement_data)
        agreement_by_condition = agreement_df.groupby("condition")["agreement_rate"].mean().reindex(conds)
        bars = ax3.bar(np.arange(len(agreement_by_condition)), agreement_by_condition.values, alpha=0.7)
        ax3.set_title('Agent Agreement Rate by Configuration', fontweight='bold')
        ax3.set_ylabel('Agreement Rate')
        ax3.set_ylim(0, 1)
        ax3.set_xticks(np.arange(len(agreement_by_condition)))
        ax3.set_xticklabels([_display_condition_name(c) for c in conds])
        _tilt_xlabels(ax3, rotation=0)
        # Add value labels
        for bar, value in zip(bars, agreement_by_condition.values):
            ax3.text(bar.get_x() + bar.get_width()/2., value + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Configuration effectiveness heatmap
    ax4 = axes[1, 1]
    df["dataset_display"] = df["dataset"].map(_display_dataset_name)
    heatmap_data = df.groupby(["condition", "dataset_display"])["majority_is_correct"].mean().unstack()
    if not heatmap_data.empty:
        heatmap_data = heatmap_data.rename(index={c: _display_condition_name(str(c)) for c in heatmap_data.index})
        from figure_style import get_colormaps
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap=get_colormaps().sequential, ax=ax4, annot_kws={"size": 10})
        ax4.set_title('Configuration Effectiveness Heatmap', fontweight='bold')
    
    if handles and labels_:
        fig.legend(handles, labels_, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=True)

    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.93))
    save_plot(output_dir, "exp3_majority_vs_individual_analysis.pdf")

def create_detailed_agent_analysis(data: List[Dict], output_dir: str = "out/figures/exp3"):
    """Create detailed analysis of individual agent behavior."""
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No data available for detailed agent analysis")
        return
    
    # Create the plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Plot 1: Agent performance by configuration
    ax1 = axes[0, 0]
    agent_config_accuracy = df.groupby(['agent', 'condition'])['agent_is_correct'].mean().unstack()
    agent_config_accuracy.plot(kind='bar', ax=ax1, alpha=0.7)
    ax1.set_title('Agent Performance by Configuration', fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.legend(title='Configuration')
    _tilt_xlabels(ax1, rotation=45)
    
    # Plot 2: Agent performance by dataset
    ax2 = axes[0, 1]
    df["dataset_display"] = df["dataset"].map(_display_dataset_name)
    agent_dataset_accuracy = df.groupby(['agent', 'dataset_display'])['agent_is_correct'].mean().unstack()
    agent_dataset_accuracy.plot(kind='bar', ax=ax2, alpha=0.7)
    ax2.set_title('Agent Performance by Dataset', fontweight='bold')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    ax2.legend(title='Dataset')
    _tilt_xlabels(ax2, rotation=45)
    
    # Plot 3: Agent performance evolution
    ax3 = axes[0, 2]
    agent_turn_accuracy = df.groupby(['agent', 'turn'])['agent_is_correct'].mean().unstack()
    for agent in agent_turn_accuracy.index:
        ax3.plot(agent_turn_accuracy.columns, agent_turn_accuracy.loc[agent], 
                marker='o', label=agent, linewidth=2)
    ax3.set_title('Agent Performance Evolution', fontweight='bold')
    ax3.set_xlabel('Turn')
    ax3.set_ylabel('Accuracy')
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Agent agreement patterns
    ax4 = axes[1, 0]
    # Calculate how often each agent agrees with the majority
    agreement_data = []
    for condition in df['condition'].unique():
        condition_data = df[df['condition'] == condition]
        for turn in condition_data['turn'].unique():
            turn_data = condition_data[condition_data['turn'] == turn]
            if len(turn_data) >= 5:
                majority_correct = turn_data['majority_is_correct'].iloc[0]
                for _, agent_data in turn_data.iterrows():
                    agreement_data.append({
                        'agent': agent_data['agent'],
                        'condition': condition,
                        'agrees_with_majority': agent_data['agent_is_correct'] == majority_correct
                    })
    
    if agreement_data:
        agreement_df = pd.DataFrame(agreement_data)
        agent_agreement = agreement_df.groupby('agent')['agrees_with_majority'].mean()
        bars = ax4.bar(agent_agreement.index, agent_agreement.values, alpha=0.7)
        ax4.set_title('Agent Agreement with Majority', fontweight='bold')
        ax4.set_ylabel('Agreement Rate')
        ax4.set_ylim(0, 1)
        _tilt_xlabels(ax4, rotation=45)
        # Add value labels
        for bar, value in zip(bars, agent_agreement.values):
            ax4.text(bar.get_x() + bar.get_width()/2., value + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: Configuration comparison heatmap
    ax5 = axes[1, 1]
    config_comparison = df.groupby(['condition', 'dataset'])['majority_is_correct'].mean().unstack()
    if not config_comparison.empty:
        from figure_style import get_colormaps
        sns.heatmap(config_comparison, annot=True, fmt=".3f", cmap=get_colormaps().sequential, ax=ax5)
        ax5.set_title('Configuration Performance Heatmap', fontweight='bold')
    
    # Plot 6: Statistical summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate summary statistics
    total_samples = len(df.groupby(['dataset', 'condition']).size())
    avg_majority_accuracy = df['majority_is_correct'].mean()
    avg_individual_accuracy = df['agent_is_correct'].mean()
    improvement = avg_majority_accuracy - avg_individual_accuracy
    
    summary_text = f"""
    Statistical Summary:
    
    Total Samples: {total_samples}
    Average Majority Accuracy: {avg_majority_accuracy:.3f}
    Average Individual Accuracy: {avg_individual_accuracy:.3f}
    Overall Improvement: {improvement:.3f}
    
    Configurations: {len(df['condition'].unique())}
    Datasets: {len(df['dataset'].unique())}
    Agents: {len(df['agent'].unique())}
    Max Turns: {df['turn'].max() if 'turn' in df.columns else 'N/A'}
    """
    
    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    save_plot(output_dir, "exp3_detailed_agent_analysis.pdf")

def create_accuracy_heatmaps(data: List[Dict], output_dir: str = "out/figures/exp3"):
    """Create accuracy heatmaps for each experimental condition showing turn vs agent performance."""
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No data available for accuracy heatmaps")
        return
    
    # Get unique conditions (stable ordering)
    conditions = sorted(df["condition"].astype(str).unique().tolist(), key=_condition_sort_key)

    # With larger fonts, a single-row grid becomes unreadable. Use a compact 2×3 layout (or similar).
    n = len(conditions)
    ncols = min(3, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 5.4 * nrows))
    axes_list = np.array(axes).reshape(-1).tolist()
    
    for i, condition in enumerate(conditions):
        ax = axes_list[i]
        
        # Filter data for this condition
        condition_data = df[df['condition'] == condition]
        
        # Create pivot table for heatmap: turn vs agent
        heatmap_data = condition_data.pivot_table(
            values='agent_is_correct',
            index='turn',
            columns='agent',
            aggfunc='mean'
        )
        
        # Create heatmap
        if not heatmap_data.empty:
            from figure_style import get_colormaps
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=".3f",
                cmap=get_colormaps().sequential,
                ax=ax,
                cbar_kws={"label": "Accuracy"},
                annot_kws={"size": 10},
            )
            ax.set_title(_display_condition_name(str(condition)), fontweight="bold")
            ax.set_xlabel("Agent", fontweight="bold")
            ax.set_ylabel("Turn", fontweight="bold")
            _tilt_xlabels(ax, rotation=35)
            
            # Add condition description
            if condition == "baseline":
                description = "5 Normal Agents"
            elif condition == "misinformed":
                description = "1 Misinformed + 4 Normal Agents"
            elif condition == "irrelevant_misinformed":
                description = "4 Misinformed + 1 Normal Agent"
            else:
                description = condition
            
            ax.text(
                0.5,
                -0.17,
                description,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
            )
        else:
            ax.text(0.5, 0.5, f'No data for {condition}', transform=ax.transAxes,
                   ha='center', va='center', fontsize=13)
            ax.set_title(f'No Data: {condition}', fontweight='bold')

    # Hide unused axes, if any.
    for j in range(len(conditions), len(axes_list)):
        axes_list[j].axis("off")

    fig.suptitle("Accuracy Heatmaps by Agent Configuration", y=1.02, fontweight="bold")
    plt.tight_layout()
    save_plot(output_dir, "exp3_accuracy_heatmaps.pdf")


def create_last_agent_analysis(last_agent_data: List[Dict], output_dir: str = "out/figures/exp3"):
    """Analyze accuracy of the *last speaking agent per turn* (ignores majority vote)."""
    df = pd.DataFrame(last_agent_data)
    if df.empty:
        print("No data available for last-agent analysis")
        return

    # Plot 1: accuracy by condition
    cond_acc = df.groupby("condition")["last_agent_is_correct"].mean()
    conds = sorted(cond_acc.index.astype(str).tolist(), key=_condition_sort_key)
    cond_acc = cond_acc.reindex(conds)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(conds))
    bars = ax.bar(x, cond_acc.values, alpha=0.8)
    ax.set_title("Last Speaking Agent (Per Turn) Accuracy by Condition", fontweight="bold")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([_display_condition_name(c) for c in conds])
    _tilt_xlabels(ax, rotation=0)
    for bar, v in zip(bars, cond_acc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, min(0.98, float(v) + 0.02), f"{float(v):.3f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    save_plot(output_dir, "exp3_last_agent_accuracy_by_condition.pdf")

    # Plot 2: accuracy by dataset x condition
    df["dataset_display"] = df["dataset"].map(_display_dataset_name)
    datasets = sorted(df["dataset_display"].astype(str).unique().tolist())
    if len(datasets) <= 1:
        # With a single dataset, the dataset×condition grouped plot is redundant and often unreadable.
        # Re-render a condition-only plot labeled with the dataset name for a clean, paper-ready figure.
        ds_title = datasets[0] if datasets else "Dataset"
        cond_acc = df.groupby("condition")["last_agent_is_correct"].mean()
        conds = sorted(cond_acc.index.astype(str).tolist(), key=_condition_sort_key)
        vals = [float(cond_acc.loc[c]) for c in conds]
        fig, ax = plt.subplots(figsize=(10.8, 5.6))
        x = np.arange(len(conds))
        bars = ax.bar(x, vals, alpha=0.82)
        ax.set_title(f"Last Speaking Agent (Per Turn) Accuracy by Condition ({ds_title})", fontweight="bold")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels([_display_condition_name(c) for c in conds])
        _tilt_xlabels(ax, rotation=0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, min(0.98, float(v) + 0.02), f"{float(v):.3f}", ha="center", va="bottom", fontweight="bold")
        plt.tight_layout()
        save_plot(output_dir, "exp3_last_agent_accuracy_by_dataset_condition.pdf")
    else:
        dc = df.groupby(["dataset_display", "condition"])["last_agent_is_correct"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(13.5, 6.4))
        sns.barplot(data=dc, x="dataset_display", y="last_agent_is_correct", hue="condition", ax=ax, alpha=0.9)
        ax.set_title("Last Speaking Agent (Per Turn) Accuracy by Dataset and Condition", fontweight="bold")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.legend(title="Condition", loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=True)
        _tilt_xlabels(ax, rotation=35)
        plt.tight_layout()
        save_plot(output_dir, "exp3_last_agent_accuracy_by_dataset_condition.pdf")


def create_vote_correctness_analysis(vote_data: List[Dict], output_dir: str = "out/figures/exp3"):
    """Assess which agents vote for correct vs wrong answers."""
    df = pd.DataFrame(vote_data)
    if df.empty:
        print("No data available for vote correctness analysis")
        return

    # Overall correctness rate per voter
    stats = (
        df.groupby("voter", observed=True)["voted_is_correct"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "correct_rate", "count": "n"})
        .sort_values("correct_rate", ascending=False)
    )
    stats["wrong_rate"] = 1.0 - stats["correct_rate"]

    # Heatmap: voter x condition
    heat = df.groupby(["voter", "condition"], observed=True)["voted_is_correct"].mean().unstack("condition")

    fig_h = max(6.2, 0.55 * max(len(stats), 8))
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(20.0, fig_h),
        gridspec_kw={"width_ratios": [1.05, 1.35]},
    )

    # Stacked horizontal bars: correct vs wrong
    y = np.arange(len(stats))
    from figure_style import get_palette
    pal6 = get_palette(6)
    ax1.barh(y, stats["correct_rate"], color=pal6[2], alpha=0.85, label="voted correct")  # sage
    ax1.barh(y, stats["wrong_rate"], left=stats["correct_rate"], color=pal6[3], alpha=0.75, label="voted wrong")  # rose
    ax1.set_yticks(y)
    ax1.set_yticklabels(stats["voter"].astype(str).tolist())
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("Share of votes")
    ax1.set_title("Vote Correctness by Agent", fontweight="bold")
    ax1.invert_yaxis()
    ax1.legend(loc="lower right", frameon=True)
    for i, row in stats.reset_index(drop=True).iterrows():
        ax1.text(0.98, i, f"{row['correct_rate']:.3f} (n={int(row['n'])})", ha="right", va="center", fontsize=11)

    # Heatmap
    if heat is not None and not heat.empty:
        heat = heat.rename(columns={c: _display_condition_name(str(c)) for c in heat.columns})
        from figure_style import get_colormaps
        sns.heatmap(
            heat,
            annot=True,
            fmt=".3f",
            cmap=get_colormaps().sequential,
            vmin=0,
            vmax=1,
            ax=ax2,
            cbar_kws={"label": "P(vote correct)"},
            annot_kws={"size": 10},
        )
        ax2.set_title("Vote Correctness by Condition", fontweight="bold")
        ax2.set_xlabel("Condition")
        ax2.set_ylabel("")
        _tilt_xlabels(ax2, rotation=25)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "No condition-level vote data", ha="center", va="center")

    fig.suptitle("Which Agents Vote Correct vs Wrong?", y=1.02, fontweight="bold")
    plt.tight_layout()
    save_plot(output_dir, "exp3_vote_correctness_analysis.pdf")


def create_last_agent_vote_analysis(last_agent_vote_data: List[Dict], output_dir: str = "out/figures/exp3"):
    """Analyze correctness of the vote cast by the last speaking agent (per voting turn)."""
    df = pd.DataFrame(last_agent_vote_data)
    if df.empty:
        print("No data available for last-agent vote analysis")
        return

    # Plot 1: vote accuracy by condition
    cond_acc = df.groupby("condition")["last_voter_voted_is_correct"].mean()
    conds = sorted(cond_acc.index.astype(str).tolist(), key=_condition_sort_key)
    cond_acc = cond_acc.reindex(conds)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(conds))
    bars = ax.bar(x, cond_acc.values, alpha=0.8)
    ax.set_title("Last Speaking Agent (Per Voting Turn) Vote Accuracy by Condition", fontweight="bold")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Vote accuracy")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([_display_condition_name(c) for c in conds])
    _tilt_xlabels(ax, rotation=0)
    for bar, v in zip(bars, cond_acc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, min(0.98, float(v) + 0.02), f"{float(v):.3f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    save_plot(output_dir, "exp3_last_agent_vote_accuracy_by_condition.pdf")

    # Plot 2: vote accuracy by dataset x condition
    df["dataset_display"] = df["dataset"].map(_display_dataset_name)
    datasets = sorted(df["dataset_display"].astype(str).unique().tolist())
    if len(datasets) <= 1:
        ds_title = datasets[0] if datasets else "Dataset"
        cond_acc = df.groupby("condition")["last_voter_voted_is_correct"].mean()
        conds = sorted(cond_acc.index.astype(str).tolist(), key=_condition_sort_key)
        vals = [float(cond_acc.loc[c]) for c in conds]
        fig, ax = plt.subplots(figsize=(10.8, 5.6))
        x = np.arange(len(conds))
        bars = ax.bar(x, vals, alpha=0.82)
        ax.set_title(f"Last Speaking Agent (Per Voting Turn) Vote Accuracy by Condition ({ds_title})", fontweight="bold")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Vote accuracy")
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels([_display_condition_name(c) for c in conds])
        _tilt_xlabels(ax, rotation=0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, min(0.98, float(v) + 0.02), f"{float(v):.3f}", ha="center", va="bottom", fontweight="bold")
        plt.tight_layout()
        save_plot(output_dir, "exp3_last_agent_vote_accuracy_by_dataset_condition.pdf")
    else:
        dc = df.groupby(["dataset_display", "condition"])["last_voter_voted_is_correct"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(13.5, 6.4))
        sns.barplot(data=dc, x="dataset_display", y="last_voter_voted_is_correct", hue="condition", ax=ax, alpha=0.9)
        ax.set_title("Last Speaking Agent (Per Voting Turn) Vote Accuracy by Dataset and Condition", fontweight="bold")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Vote accuracy")
        ax.set_ylim(0, 1)
        ax.legend(title="Condition", loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=True)
        _tilt_xlabels(ax, rotation=35)
        plt.tight_layout()
        save_plot(output_dir, "exp3_last_agent_vote_accuracy_by_dataset_condition.pdf")


def create_last_agent_response_vs_vote_comparison(
    last_agent_data: List[Dict],
    last_agent_vote_data: List[Dict],
    output_dir: str = "out/figures/exp3",
    *,
    include_delta: bool = True,
    y_min: float = 0.0,
    output_name: str = "exp3_last_agent_response_vs_vote_accuracy.pdf",
):
    """Compare last-agent consensus accuracy vs voting accuracy by condition."""
    df_resp = pd.DataFrame(last_agent_data)
    df_vote = pd.DataFrame(last_agent_vote_data)

    if df_resp.empty:
        print("No data available for last-agent response vs vote comparison (response data empty)")
        return
    if df_vote.empty:
        print("No data available for last-agent response vs vote comparison (vote data empty)")
        return

    # If model is unavailable in legacy data, treat as a single "combined" model.
    if "model" not in df_resp.columns:
        df_resp["model"] = "combined"
    if "model" not in df_vote.columns:
        df_vote["model"] = "combined"

    resp_by_model_cond = df_resp.groupby(["model", "condition"])["last_agent_is_correct"].mean()
    vote_by_model_cond = df_vote.groupby(["model", "condition"])["last_voter_voted_is_correct"].mean()

    model_names = sorted(
        set(resp_by_model_cond.index.get_level_values(0).astype(str)).union(
            set(vote_by_model_cond.index.get_level_values(0).astype(str))
        )
    )
    conditions = sorted(
        set(resp_by_model_cond.index.get_level_values(1).astype(str)).union(
            set(vote_by_model_cond.index.get_level_values(1).astype(str))
        ),
        key=_condition_sort_key,
    )

    fig, ax = plt.subplots(figsize=(11.6, 8.7))
    x = np.arange(len(conditions))

    from figure_style import get_palette
    # Group visual identity by model: one base color per model.
    # Both metrics (Consensus/Voting/Delta) within a model share the same color
    # and are distinguished by linestyle and marker.
    if len(model_names) == 2:
        # Purple + blue-gray from the project palette. Avoid green/red shades
        # because the x-tick label tint runs green→red.
        pal4 = get_palette(4)
        model_palette = [pal4[0], pal4[1]]
    else:
        model_palette = get_palette(max(4, len(model_names)))

    def _hex_to_rgb01(hex_color: str):
        h = str(hex_color).lstrip("#")
        if len(h) != 6:
            return (0.3, 0.3, 0.3)
        return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

    if include_delta:
        metric_specs = [
            ("Consensus", "o", "solid"),
            ("Voting", "s", "dashed"),
            ("Δ (Voting − Consensus)", "^", "dotted"),
        ]
    else:
        metric_specs = [
            ("Consensus", "o", "solid"),
            ("Voting", "s", "dashed"),
        ]

    series_for_labels = []
    legend_group_entries = {}
    for model_idx, model_name in enumerate(model_names):
        resp_vals = []
        vote_vals = []
        delta_vals = []
        for c in conditions:
            resp_v = resp_by_model_cond.get((model_name, c), np.nan)
            vote_v = vote_by_model_cond.get((model_name, c), np.nan)
            resp_v = float(resp_v) if pd.notna(resp_v) else np.nan
            vote_v = float(vote_v) if pd.notna(vote_v) else np.nan
            delta_v = vote_v - resp_v if (np.isfinite(vote_v) and np.isfinite(resp_v)) else np.nan
            resp_vals.append(resp_v)
            vote_vals.append(vote_v)
            delta_vals.append(delta_v)

        model_display = _display_model_name_short(model_name)
        series_by_metric = (resp_vals, vote_vals, delta_vals) if include_delta else (resp_vals, vote_vals)

        legend_group_entries[model_display] = []
        for metric_idx, ((metric_label, marker, line_style), ys) in enumerate(
            zip(metric_specs, series_by_metric)
        ):
            base_rgb = _hex_to_rgb01(model_palette[model_idx % len(model_palette)])
            # All metrics share the model's color; linestyle + marker carry the
            # consensus/voting/delta distinction.
            line_color = base_rgb
            line_handle = ax.plot(
                x,
                ys,
                marker=marker,
                linewidth=2.8,
                linestyle=line_style,
                label="_nolegend_",
                color=line_color,
                alpha=0.95,
            )[0]
            legend_group_entries[model_display].append((metric_label, line_handle))
            series_for_labels.append((ys, marker, model_display, metric_idx, line_color))

    base_fs = float(plt.rcParams.get("font.size", 12))
    label_fs = max(16.0, base_fs * 1.15)
    x_tick_fs = max(12.0, base_fs * 0.80)
    y_tick_fs = max(14.0, base_fs * 1.0)
    legend_fs = max(14.0, base_fs * 1.0)
    num_fs = max(13.0, base_fs * 0.92)
    title_fs = max(16.0, base_fs * 1.1)

    ax.set_title(
        "Multi-Agent Accuracy (Consensus vs. Voting)",
        fontweight="bold",
        fontsize=title_fs,
        pad=14,
    )
    ax.set_xlabel("Agent Configuration", fontsize=label_fs, labelpad=18)
    ax.set_ylabel("Accuracy", fontsize=label_fs)
    ax.set_ylim(y_min, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_display_condition_name(c) for c in conditions],
        fontsize=x_tick_fs,
    )
    ax.tick_params(axis="y", labelsize=y_tick_fs)

    # Add a left→right green→red tint behind x tick labels.
    from matplotlib.colors import LinearSegmentedColormap
    from figure_style import sample_cmap
    tick_cmap = LinearSegmentedColormap.from_list("green_to_red", ["#2e7d32", "#c62828"])
    tick_colors = sample_cmap(tick_cmap, len(conditions))
    for lab, col in zip(ax.get_xticklabels(), tick_colors):
        lab.set_bbox(
            dict(
                facecolor=col,
                edgecolor="none",
                boxstyle="round,pad=0.25",
                alpha=0.25,
            )
        )
    # Labels are already multi-line; avoid rotation for readability.
    from matplotlib.lines import Line2D
    # Force model group order for stable left/right legend columns.
    ordered_model_displays = sorted(
        legend_group_entries.keys(),
        key=lambda name: (0 if "llama" in name.lower() else 1, name.lower()),
    )
    grouped_entries = []
    for model_display in ordered_model_displays:
        metric_entries = legend_group_entries[model_display]
        # Row-leading model label inside the same legend box.
        header = (Line2D([], [], linestyle="none", marker=None, color="none"), f"{model_display}:", True)
        entries = [header] + [(h, lbl, False) for lbl, h in metric_entries]
        grouped_entries.append(entries)

    # Matplotlib fills multi-column legends column-major.
    # Build entries by columns so rows render as:
    # [Model, Consensus, Voting, (Delta)]
    # for each model group.
    col_model = [g[0] for g in grouped_entries]
    col_consensus = [g[1] for g in grouped_entries]
    col_voting = [g[2] for g in grouped_entries]
    if include_delta:
        col_delta = [g[3] for g in grouped_entries]
        column_major_entries = col_model + col_consensus + col_voting + col_delta
        legend_ncol = 4
    else:
        column_major_entries = col_model + col_consensus + col_voting
        legend_ncol = 3

    legend_handles = [h for h, _txt, _is_header in column_major_entries]
    legend_labels = [_txt for _h, _txt, _is_header in column_major_entries]
    header_indices = [i for i, (_h, _txt, is_header) in enumerate(column_major_entries) if is_header]

    legend_obj = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=legend_ncol,
        frameon=True,
        fontsize=legend_fs,
        handlelength=2.4,
        handletextpad=0.5,
        labelspacing=0.35,
        columnspacing=1.5,
    )
    for idx in header_indices:
        if idx < len(legend_obj.get_texts()):
            legend_obj.get_texts()[idx].set_fontweight("normal")
    ax.grid(True, axis="y", alpha=0.25)

    # Value labels at each point. Compact offset keeps labels visually attached
    # to their lines; place GLM Consensus and Llama Voting below their lines so
    # they do not collide with neighbouring series. Labels share the colour of
    # the line they annotate.
    label_gap = 0.008
    for ys, _marker, model_disp, metric_idx, label_color in series_for_labels:
        disp_lc = str(model_disp).lower()
        place_below = ("glm" in disp_lc and metric_idx == 0) or (
            "llama" in disp_lc and metric_idx == 1
        )
        for xi, yi in zip(x, ys):
            if not np.isfinite(yi):
                continue
            if place_below:
                y_text = max(y_min + 0.002, float(yi) - label_gap)
                va = "top"
            else:
                y_text = min(1.01, float(yi) + label_gap)
                va = "bottom"
            ax.text(
                xi,
                y_text,
                f"{float(yi):.3f}",
                ha="center",
                va=va,
                fontsize=num_fs,
                color=label_color,
            )

    plt.tight_layout(rect=(0.0, 0.08, 1.0, 0.84))
    save_plot(output_dir, output_name)


def create_last_agent_response_vs_vote_comparison_no_delta(
    last_agent_data: List[Dict],
    last_agent_vote_data: List[Dict],
    output_dir: str = "out/figures/exp3",
):
    """Consensus vs voting only, y-axis from 0.7 (no delta series)."""
    create_last_agent_response_vs_vote_comparison(
        last_agent_data,
        last_agent_vote_data,
        output_dir,
        include_delta=False,
        y_min=0.7,
        output_name="exp3_last_agent_response_vs_vote_accuracy_no_delta.pdf",
    )


def create_last_agent_vote_minus_response_delta_plot(
    last_agent_data: List[Dict],
    last_agent_vote_data: List[Dict],
    output_dir: str = "out/figures/exp3",
):
    """Plot Δ = vote accuracy − response accuracy by condition."""
    df_resp = pd.DataFrame(last_agent_data)
    df_vote = pd.DataFrame(last_agent_vote_data)

    if df_resp.empty:
        print("No data available for Δ plot (response data empty)")
        return
    if df_vote.empty:
        print("No data available for Δ plot (vote data empty)")
        return

    resp_by_cond = df_resp.groupby("condition")["last_agent_is_correct"].mean()
    vote_by_cond = df_vote.groupby("condition")["last_voter_voted_is_correct"].mean()

    conditions = sorted(set(resp_by_cond.index.astype(str)).union(set(vote_by_cond.index.astype(str))))
    delta = np.array([float(vote_by_cond.get(c, np.nan)) - float(resp_by_cond.get(c, np.nan)) for c in conditions], dtype=float)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(conditions))
    from figure_style import get_palette
    pal6 = get_palette(6)
    colors = [pal6[2] if (np.isfinite(d) and d >= 0) else pal6[3] for d in delta]
    bars = ax.bar(x, delta, color=colors, alpha=0.85)

    ax.set_title("Exp3: Delta = Vote Accuracy - Response Accuracy (last speaking agent)", fontweight="bold")
    ax.set_ylabel("Delta accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    _tilt_xlabels(ax, rotation=35)
    ax.axhline(0, color="black", linewidth=1.2, alpha=0.6)
    ax.grid(True, axis="y", alpha=0.25)

    # Symmetric-ish y-limits around 0 (with a small cushion)
    finite = delta[np.isfinite(delta)]
    if finite.size:
        m = float(np.max(np.abs(finite)))
        ax.set_ylim(-(m + 0.05), (m + 0.05))

    for bar, d in zip(bars, delta):
        if not np.isfinite(d):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            d + (0.01 if d >= 0 else -0.01),
            f"{d:+.3f}",
            ha="center",
            va="bottom" if d >= 0 else "top",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    save_plot(output_dir, "exp3_last_agent_vote_minus_response_delta.pdf")


def create_misinformed_adjustment_plot(adjustment_data: List[Dict], output_dir: str = "out/figures/exp3"):
    """Plot: Are misinformed agents more likely to adjust with more informed agents?"""
    df = pd.DataFrame(adjustment_data)
    if df.empty:
        print("No data available for misinformed adjustment analysis")
        return

    # Keep only rows with known informed_count and started_wrong=True (conditioned adjustment rate)
    df = df[df["informed_count"].notna()]
    df["informed_count"] = df["informed_count"].astype(int)
    base = df[df["started_wrong"] == True]
    if base.empty:
        print("No started-wrong cases available for misinformed adjustment analysis")
        return

    if "model" not in base.columns:
        base["model"] = "combined"

    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import PercentFormatter
    from figure_style import sample_cmap
    from figure_style import get_palette as get_fig_palette

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    grp = (
        base.groupby(["model", "informed_count"], observed=True)["adjusted_to_correct"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values(["model", "informed_count"])
    )
    # Map informed_count -> representative misinformed_count for long-form x labels.
    mix_counts = (
        base.groupby(["informed_count", "misinformed_count"], observed=True)
        .size()
        .reset_index(name="n")
        .sort_values(["informed_count", "n"], ascending=[True, False])
    )
    informed_to_mis: Dict[int, int] = {}
    for informed, mis, _n in mix_counts[["informed_count", "misinformed_count", "n"]].itertuples(index=False):
        inf_i = int(informed)
        if inf_i not in informed_to_mis and pd.notna(mis):
            informed_to_mis[inf_i] = int(mis)

    model_order = sorted(grp["model"].astype(str).unique().tolist(), key=_model_order_key)
    xs_all = sorted(grp["informed_count"].astype(int).unique().tolist())

    axis_label_fs = 18
    line_label_fs = 14
    xtick_fs = 12
    title_fs = 19
    tick_fs = 16
    peer_label_fs = 14

    # Match requested direction for x-axis context coloring.
    grad_cmap = LinearSegmentedColormap.from_list("red_to_green_adj", ["#c62828", "#2e7d32"])
    point_colors = sample_cmap(grad_cmap, len(xs_all))

    # Match the colors used in `exp3_last_agent_response_vs_vote_accuracy.pdf`:
    # purple + blue-gray (pal4[0], pal4[1]). Avoid green/red because the x-tick
    # tint already uses a red→green gradient.
    if len(model_order) == 2:
        pal4 = get_fig_palette(4)
        palette = [pal4[0], pal4[1]]
    else:
        palette = get_fig_palette(max(2, len(model_order)))
    model_display_map = {m: _display_model_name_short(m) for m in model_order}
    model_palette = {model_display_map[m]: palette[i % len(palette)] for i, m in enumerate(model_order)}
    model_markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

    for mi, model_name in enumerate(model_order):
        mg = grp[grp["model"].astype(str) == model_name].sort_values("informed_count")
        xs = mg["informed_count"].astype(float).tolist()
        ys = mg["mean"].astype(float).tolist()
        ns = mg["count"].astype(int).tolist()
        if not xs:
            continue
        display = model_display_map[model_name]
        line_color = model_palette[display]
        ax.plot(
            xs,
            ys,
            linewidth=2.8,
            alpha=0.95,
            color=line_color,
            marker=model_markers[mi % len(model_markers)],
            markersize=7,
            label=display,
        )
        place_below = "glm" in display.lower()
        label_gap_above = 0.012
        label_gap_below = 0.04
        for x, y, n in zip(xs, ys, ns):
            if place_below:
                y_text = max(0.02, float(y) - label_gap_below)
                va = "top"
            else:
                y_text = min(0.98, float(y) + label_gap_above)
                va = "bottom"
            ax.text(
                float(x),
                y_text,
                f"{float(y) * 100:.1f}%\n(n={int(n)})",
                ha="center",
                va=va,
                fontsize=line_label_fs,
                fontweight="normal",
                color=line_color,
            )

    # Keep x-position color hints for informed-agent counts.
    ax.scatter(xs_all, [0.0] * len(xs_all), s=0.1, c=point_colors, alpha=0.0)  # anchor colors
    ax.set_title("Convincing Misinformed Agents", fontweight="bold", fontsize=title_fs)
    ax.set_xlabel("Agent Configuration", fontsize=axis_label_fs, labelpad=10)
    ax.set_ylabel("P(adjusted | misinformed)", fontsize=axis_label_fs)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_yticks(np.arange(0.0, 1.01, 0.25))
    tick_vals = xs_all
    ax.set_xticks(tick_vals)
    tick_labels = []
    for inf in tick_vals:
        mis = informed_to_mis.get(int(inf))
        if mis is None:
            tick_labels.append(str(inf))
        else:
            # Swapped emphasis for this plot: uninformed first, misinformed second.
            tick_labels.append(f"{int(inf)} uninformed\n{int(mis)} misinformed")
    ax.set_xticklabels(tick_labels)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.tick_params(axis="x", labelsize=xtick_fs)

    # Reference marker for the 3-uninformed / 2-misinformed configuration.
    if 3 in tick_vals:
        ax.axvline(x=3, linestyle=":", linewidth=2.0, color=get_named_colors().get("slate", "#4A4A4A"), alpha=0.65)
        ax.text(
            3.06,
            0.5,
            "peer pressure",
            ha="left",
            va="center",
            fontsize=peer_label_fs,
            color=get_named_colors().get("slate", "#4A4A4A"),
            fontstyle="italic",
        )
    for lab, col in zip(ax.get_xticklabels(), point_colors):
        lab.set_bbox(
            dict(
                facecolor=col,
                edgecolor="none",
                boxstyle="round,pad=0.25",
                alpha=0.22,
            )
        )
    if len(model_order) > 1:
        handles, labels = ax.get_legend_handles_labels()
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.915),
            ncol=min(2, len(labels)),
            frameon=True,
            fontsize=14,
        )
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    else:
        plt.tight_layout()
    save_plot(output_dir, "exp3_misinformed_adjustment_vs_informed_count.pdf")


def create_model_comparison_plot(data: List[Dict], output_dir: str = "out/figures/exp3"):
    """Combined-model comparison based on majority correctness."""
    df = pd.DataFrame(data)
    if df.empty or "model" not in df.columns:
        return
    stats = (
        df.groupby("model", observed=True)["majority_is_correct"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n"})
        .sort_values("accuracy", ascending=False)
    )
    if len(stats) < 2:
        return

    from shared_visualization import get_palette

    pal = get_palette()
    color_map = {m: pal[i % len(pal)] for i, m in enumerate(stats["model"].tolist())}
    stats = stats.sort_values("model", key=lambda s: s.map(lambda m: _model_order_key(str(m))))
    stats["model_display"] = stats["model"].astype(str).map(_display_model_name_short)

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    sns.barplot(
        data=stats,
        x="model_display",
        y="accuracy",
        hue="model",
        hue_order=stats["model"].tolist(),
        dodge=False,
        palette=color_map,
        legend=False,
        ax=ax,
    )
    ax.set_title("Exp3 Model Comparison (Majority Accuracy)", fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    for i, row in stats.reset_index(drop=True).iterrows():
        ax.text(i, min(0.98, float(row["accuracy"]) + 0.02), f"n={int(row['n'])}", ha="center", va="bottom")
    for lab in ax.get_xticklabels():
        lab.set_rotation(20)
        lab.set_ha("right")
        lab.set_rotation_mode("anchor")
    plt.tight_layout()
    save_plot(output_dir, "exp3_model_comparison_overall_accuracy.pdf")


def create_model_comparison_by_condition_plot(data: List[Dict], output_dir: str = "out/figures/exp3"):
    """Combined-model comparison by agent configuration condition."""
    df = pd.DataFrame(data)
    if df.empty or "model" not in df.columns or "condition" not in df.columns:
        return

    models = sorted(df["model"].dropna().astype(str).unique().tolist(), key=_model_order_key)
    if len(models) < 2:
        return

    stats = (
        df.groupby(["condition", "model"], observed=True)["majority_is_correct"]
        .mean()
        .reset_index()
        .rename(columns={"majority_is_correct": "accuracy"})
    )
    if stats.empty:
        return

    conds = sorted(stats["condition"].astype(str).unique().tolist(), key=_condition_sort_key)
    conds_display = [_display_condition_name(c) for c in conds]
    stats["condition_display"] = stats["condition"].astype(str).map(_display_condition_name)
    stats["model_display"] = stats["model"].astype(str).map(_display_model_name_short)
    model_display = [_display_model_name_short(m) for m in models]

    from shared_visualization import get_palette

    pal = get_palette()
    model_palette = {m: pal[i % len(pal)] for i, m in enumerate(model_display)}

    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    sns.barplot(
        data=stats,
        x="condition_display",
        y="accuracy",
        hue="model_display",
        order=conds_display,
        hue_order=model_display,
        palette=model_palette,
        ax=ax,
    )
    ax.set_title("Model Comparison by Agent Configuration", fontweight="bold")
    ax.set_xlabel("Agent Configuration")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    _tilt_xlabels(ax, rotation=0)
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=max(1, len(labels)), frameon=True)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    save_plot(output_dir, "exp3_model_comparison_by_condition.pdf")


def main():
    """Main function to generate exp3 visualizations."""
    
    parser = argparse.ArgumentParser(description="Create visualizations for exp3 results")
    parser.add_argument("--results_file", type=str, default=None,
                       help="Path to the results JSON file (if not specified, will find all exp3 results)")
    parser.add_argument("--output_dir", type=str, default="out/figures/exp3",
                       help="Output directory for figures")
    parser.add_argument("--all_plots", action="store_true",
                       help="Generate all plots")
    parser.add_argument("--agent_configuration", action="store_true",
                       help="Generate agent configuration analysis plot")
    parser.add_argument("--majority_individual", action="store_true",
                       help="Generate majority vs individual analysis plot")
    parser.add_argument("--detailed_agent", action="store_true",
                       help="Generate detailed agent analysis plot")
    parser.add_argument("--accuracy_heatmaps", action="store_true",
                       help="Generate accuracy heatmaps for each condition")
    parser.add_argument("--last_agent", action="store_true",
                       help="Generate last-speaking-agent accuracy analysis (ignores majority vote)")
    parser.add_argument("--last_agent_vote", action="store_true",
                       help="Generate last-speaking-agent vote accuracy analysis (uses votesEachTurn)")
    parser.add_argument("--last_agent_compare", action="store_true",
                       help="Generate comparison plot of last-agent response vs vote accuracy")
    parser.add_argument("--misinfo_adjustment", action="store_true",
                       help="Plot whether misinformed agents adjust more with more informed agents")
    parser.add_argument("--vote_analysis", action="store_true",
                       help="Generate per-agent vote correctness analysis (who votes correct vs wrong)")
    
    args = parser.parse_args()
    if not any(
        [
            args.all_plots,
            args.agent_configuration,
            args.majority_individual,
            args.detailed_agent,
            args.accuracy_heatmaps,
            args.last_agent,
            args.last_agent_vote,
            args.last_agent_compare,
            args.misinfo_adjustment,
            args.vote_analysis,
        ]
    ):
        args.all_plots = True
    
    # Load results
    if args.results_file:
        # Load specific results file
        print(f"Loading results from {args.results_file}")
        try:
            results = load_results(args.results_file)
        except FileNotFoundError:
            print(f"Results file {args.results_file} not found. Please run the experiment first.")
            return
        
        # Derive dataset name from filename when possible (exp3_results_{dataset}.json)
        ds_guess = None
        model_guess = infer_model_from_result_path(args.results_file, out_dir="out")
        try:
            base = os.path.basename(args.results_file)
            if base.startswith("exp3_results_") and base.endswith(".json"):
                ds_guess = base.replace("exp3_results_", "").replace(".json", "")
        except Exception:
            ds_guess = None

        # Extract data
        print("Extracting agent solutions data...")
        data = extract_agent_solutions_data(results, dataset_name=ds_guess, model_name=model_guess)
        print("Extracting last-agent data...")
        last_agent_data = extract_last_agent_data(results, dataset_name=ds_guess, model_name=model_guess)
        print("Extracting last-agent vote data...")
        last_agent_vote_data = extract_last_agent_vote_data(results, dataset_name=ds_guess, model_name=model_guess)
        print("Extracting misinformed adjustment data...")
        misinfo_adjustment_data = extract_misinformed_adjustment_data(
            results, dataset_name=ds_guess, model_name=model_guess
        )
        print("Extracting vote data...")
        vote_data = extract_vote_data(results, dataset_name=ds_guess, model_name=model_guess)
        
        if not data:
            print("No agent solutions data found.")
            return
        
        print(f"Extracted {len(data)} data points")
    else:
        # Find and load all exp3 results files
        print("No specific results file provided. Finding all exp3 results files...")
        
        # Find all exp3 results files across model subfolders.
        exp3_files = find_result_files(out_dir="out", prefix="exp3_results_")
        
        if not exp3_files:
            print("No exp3 results files found in the 'out' directory.")
            print("Please run exp3.py first to generate results.")
            return
        
        print(f"Found {len(exp3_files)} exp3 results files:")
        for file in exp3_files:
            print(f"  - {file}")
        
        # Load and combine all results
        all_data = []
        all_last_agent_data = []
        all_last_agent_vote_data = []
        all_misinfo_adjustment_data = []
        all_vote_data = []
        for file_path in exp3_files:
            print(f"\nLoading {file_path}...")
            try:
                results = load_results(file_path)
                # Derive dataset/model from path.
                base = os.path.basename(file_path)
                ds = base.replace("exp3_results_", "").replace(".json", "")
                model = infer_model_from_result_path(file_path, out_dir="out")
                file_data = extract_agent_solutions_data(results, dataset_name=ds, model_name=model)
                file_last = extract_last_agent_data(results, dataset_name=ds, model_name=model)
                file_last_vote = extract_last_agent_vote_data(results, dataset_name=ds, model_name=model)
                file_misinfo_adj = extract_misinformed_adjustment_data(results, dataset_name=ds, model_name=model)
                file_votes = extract_vote_data(results, dataset_name=ds, model_name=model)
                if file_data:
                    all_data.extend(file_data)
                    print(f"  Extracted {len(file_data)} data points")
                else:
                    print(f"  No agent solutions data found in {file_path}")
                if file_last:
                    all_last_agent_data.extend(file_last)
                if file_last_vote:
                    all_last_agent_vote_data.extend(file_last_vote)
                if file_misinfo_adj:
                    all_misinfo_adjustment_data.extend(file_misinfo_adj)
                if file_votes:
                    all_vote_data.extend(file_votes)
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
        
        if not all_data:
            print("No agent solutions data found in any exp3 results files.")
            print("Please run exp3.py first to generate results with agent solutions.")
            return
        
        data = all_data
        last_agent_data = all_last_agent_data
        last_agent_vote_data = all_last_agent_vote_data
        misinfo_adjustment_data = all_misinfo_adjustment_data
        vote_data = all_vote_data
        print(f"\nTotal data points extracted: {len(data)}")
    
    # Create output directories.
    os.makedirs(args.output_dir, exist_ok=True)
    combined_dir = os.path.join(args.output_dir, "combined")
    model_root = os.path.join(args.output_dir, "models")
    os.makedirs(combined_dir, exist_ok=True)
    os.makedirs(model_root, exist_ok=True)

    def _run_selected_plots(
        out_dir: str,
        d: List[Dict],
        d_last: List[Dict],
        d_last_vote: List[Dict],
        d_adjust: List[Dict],
        d_vote: List[Dict],
    ) -> None:
        if args.all_plots or args.agent_configuration:
            create_agent_configuration_analysis(d, out_dir)
        if args.all_plots or args.majority_individual:
            create_majority_vs_individual_analysis(d, out_dir)
        if args.all_plots or args.detailed_agent:
            create_detailed_agent_analysis(d, out_dir)
        if args.all_plots or args.accuracy_heatmaps:
            create_accuracy_heatmaps(d, out_dir)
        if args.all_plots or args.last_agent:
            create_last_agent_analysis(d_last, out_dir)
        if args.all_plots or args.last_agent_vote:
            create_last_agent_vote_analysis(d_last_vote, out_dir)
        if args.all_plots or args.last_agent_compare:
            create_last_agent_response_vs_vote_comparison(d_last, d_last_vote, out_dir)
            create_last_agent_response_vs_vote_comparison_no_delta(d_last, d_last_vote, out_dir)
        if args.all_plots or args.misinfo_adjustment:
            create_misinformed_adjustment_plot(d_adjust, out_dir)
        if args.all_plots or args.vote_analysis:
            create_vote_correctness_analysis(d_vote, out_dir)

    print("Creating combined exp3 figures...")
    _run_selected_plots(combined_dir, data, last_agent_data, last_agent_vote_data, misinfo_adjustment_data, vote_data)
    create_model_comparison_plot(data, combined_dir)
    create_model_comparison_by_condition_plot(data, combined_dir)

    # Per-model figures.
    models = sorted({str(r.get("model", "combined")) for r in data if isinstance(r, dict)})
    for model in models:
        model_dir = os.path.join(model_root, model)
        os.makedirs(model_dir, exist_ok=True)
        data_m = [r for r in data if str(r.get("model", "combined")) == model]
        last_m = [r for r in last_agent_data if str(r.get("model", "combined")) == model]
        last_vote_m = [r for r in last_agent_vote_data if str(r.get("model", "combined")) == model]
        adjust_m = [r for r in misinfo_adjustment_data if str(r.get("model", "combined")) == model]
        vote_m = [r for r in vote_data if str(r.get("model", "combined")) == model]
        if not data_m:
            continue
        print(f"Creating per-model exp3 figures for {model}...")
        _run_selected_plots(model_dir, data_m, last_m, last_vote_m, adjust_m, vote_m)

    print(f"All plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
