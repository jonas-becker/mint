#!/usr/bin/env python3

"""
Shared utilities for experiment scripts and visualization scripts.
This module contains common functions, classes, and constants used across
exp1.py, exp2.py, exp1_figures.py, and exp2_figures.py.
"""

import json
import torch
import argparse
import re
import random
import os
import warnings
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Suppress warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast")

# Model configuration
MODEL_NAME = "openai/gpt-oss-120b"
GLM47_FLASH_MODEL_NAME = "zai-org/GLM-4.7-Flash"

# Prompt templates
PROMPT_BASELINE = """
{sentence}

{options_text}

Solve the task.
"""

PROMPT_MISINFORMED = """
{sentence}

{options_text}

You have this extra information: {misinformation}

Solve the task.
"""

# Special prompt templates for complex_web_questions (free-form questions)
PROMPT_BASELINE_FREE_FORM = """
{sentence}

Provide a clear and accurate answer to the question above.
"""

PROMPT_MISINFORMED_FREE_FORM = """
{sentence}

You have this extra information: {misinformation}

Provide a clear and accurate answer to the question above.
"""

def build_debate_system_prompt(sentence: str, options_text: str, is_multiple_choice: bool, misinformation: str = None) -> str:
    """Build a unified system prompt for debate agents across experiments.

    - For multiple choice datasets, include the options and require an explicit final choice.
    - For free-form datasets, ask for a clear and accurate answer.
    - Optionally include misinformation text.
    """
    if is_multiple_choice:
        base = f"{sentence}\n\n{options_text}\n\n"
        task_instruction = (
            "You are an agent in a debate about solving this task. You are thoughtful and analytical.\n\n"
            "Your role is to:\n"
            "1. Carefully analyze the given task\n"
            "2. Present your reasoning clearly and logically\n"
            "3. Respond to counterarguments respectfully\n"
            "4. Be open to changing your mind if presented with compelling evidence\n"
            "5. Always provide your final answer choice (A, B, C, etc.) at the end of your response\n\n"
            "Solve the task."
        )
    else:
        base = f"{sentence}\n\nProvide a clear and accurate answer to the question above.\n\n"
        task_instruction = (
            "You are an agent in a debate about solving this task. You are thoughtful and analytical.\n\n"
            "Your role is to:\n"
            "1. Carefully analyze the given task\n"
            "2. Present your reasoning clearly and logically\n"
            "3. Respond to counterarguments respectfully\n"
            "4. Be open to changing your mind if presented with compelling evidence\n"
            "5. Always provide your final answer after the text \"Final Answer:\" at the end of your response\n\n"
            "Solve the task."
        )

    if misinformation:
        misinfo_line = f"You have this extra information: {misinformation}\n\n"
    else:
        misinfo_line = ""

    return base + misinfo_line + task_instruction

class DummyModel:
    """Dummy model for testing without loading actual models."""
    def __init__(self):
        self.device = "cpu"
    
    def generate(self, input_ids, attention_mask=None, max_new_tokens=512, **kwargs):
        # Return dummy responses based on input length
        batch_size = input_ids.shape[0]
        dummy_responses = []
        
        for i in range(batch_size):
            # Generate a dummy response that looks like a real model response
            if i % 3 == 0:
                response = "Based on my analysis, the answer is A) First option."
            elif i % 3 == 1:
                response = "After careful consideration, I believe the correct answer is B) Second option."
            else:
                response = "Looking at the evidence, the answer appears to be C) Third option."
            
            # Pad with dummy tokens to simulate max_new_tokens
            dummy_tokens = torch.randint(0, 1000, (len(response.split()) + 10,))
            dummy_responses.append(dummy_tokens)
        
        # Stack responses
        max_len = max(len(r) for r in dummy_responses)
        padded_responses = []
        for response in dummy_responses:
            if len(response) < max_len:
                padding = torch.zeros(max_len - len(response), dtype=response.dtype)
                response = torch.cat([response, padding])
            padded_responses.append(response)
        
        return torch.stack(padded_responses)

class DummyTokenizer:
    """Dummy tokenizer for testing without loading actual models."""
    def __init__(self):
        self.eos_token_id = 2
        self.pad_token_id = 0
    
    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=512):
        # Create dummy tokenized output
        batch_size = len(texts)
        max_len = min(max_length, 100)  # Cap at 100 tokens
        
        input_ids = torch.randint(0, 1000, (batch_size, max_len))
        attention_mask = torch.ones((batch_size, max_len))
        
        return type('obj', (object,), {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })()
    
    def decode(self, tokens, skip_special_tokens=True):
        # Return a dummy decoded response that might match some expected answers
        dummy_responses = [
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
            "The region speaks Albanian and Greek languages.",
            "Based on my analysis, the answer is A) First option.",
            "After careful consideration, I believe the correct answer is B) Second option.",
            "Looking at the evidence, the answer appears to be C) Third option."
        ]
        return random.choice(dummy_responses)
    
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # Dummy chat template
        formatted = ""
        for message in messages:
            if message["role"] == "system":
                formatted += f"<|system|>\n{message['content']}<|end|>\n"
            elif message["role"] == "user":
                formatted += f"<|user|>\n{message['content']}<|end|>\n"
            elif message["role"] == "assistant":
                formatted += f"<|assistant|>\n{message['content']}<|end|>\n"
        return formatted

def build_common_arg_parser(description: str = "Run experiment") -> argparse.ArgumentParser:
    """Build ArgumentParser with flags shared by experiment scripts (exp1, exp1a, ...)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--max_samples", type=int, default=None, 
                       help="Maximum number of samples to evaluate per dataset")
    parser.add_argument(
        "--model_name",
        "--model",
        dest="model_name",
        type=str,
        default=MODEL_NAME,
        help=(
            "Model name to use for evaluation "
            f"(e.g. {GLM47_FLASH_MODEL_NAME})"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for processing (default: 8)")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit quantization")
    parser.add_argument("--use_smaller_model", action="store_true", help="Use a smaller model for testing (Llama-3.1-8B)")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate per prompt")
    parser.add_argument("--continue", action="store_true", help="Continue from previous runs (don't clear existing files)")
    parser.add_argument("--dummy_test", action="store_true", help="Run in dummy test mode without loading models")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=(
            "Comma-separated datasets to run, e.g. "
            "'winogrande,ethics,complex_web_questions'. "
            "If omitted, run all available datasets."
        ),
    )
    return parser


def get_common_args(description: str = "Run experiment") -> argparse.Namespace:
    """Get common command line arguments for experiment scripts."""
    return build_common_arg_parser(description).parse_args()


def model_name_to_safe_path(model_name: str) -> str:
    """Create a filesystem-safe token from a model name."""
    safe = (model_name or "").strip().replace("/", "_")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", safe)
    return safe or "unknown_model"

def load_dataset(file_path: str) -> List[Dict]:
    """Load dataset from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_options(options: List[str]) -> str:
    """Format options as A) B) C) etc."""
    if not options:
        return ""
    formatted = []
    for i, option in enumerate(options):
        formatted.append(f"{chr(65 + i)}) {option}")
    return "Answer Choices:\n" + "\n".join(formatted)

def extract_answer_from_response(response: str, options: List[str]) -> Optional[str]:
    """Extract the predicted answer from model response."""
    if not isinstance(response, str):
        try:
            response = str(response)
        except Exception:
            return None
    
    # Strategy 1: Look for explicit answer patterns like "The correct answer is A)" or "The answer is B)"
    explicit_patterns = [
        # Allow formats with or without trailing ')', and allow end punctuation.
        r'the correct answer is\s*([a-e])\b',
        r'the answer is\s*([a-e])\b',
        r'answer\s*[:\-]\s*([a-e])\b',
        r'final answer\s*[:\-]\s*([a-e])\b',
        r'best answer is\s*([a-e])\b',
        r'correct answer\s*[:\-]\s*([a-e])\b',
        r'my answer is\s*([a-e])\b',
        r'chosen answer\s*[:\-]\s*([a-e])\b',
        r'selected answer\s*[:\-]\s*([a-e])\b',
        r'\\boxed\{\s*([a-e])\s*\}',
    ]
    
    for pattern in explicit_patterns:
        # These explicit patterns include "answer" phrases and are safe to search across
        # the whole response (the option listing itself won't contain them).
        matches = re.findall(pattern, response, flags=re.IGNORECASE)
        if matches:
            answer_letter = matches[-1].upper()  # Take the last match and convert to uppercase
            answer_index = ord(answer_letter) - ord('A')
            if 0 <= answer_index < len(options):
                return options[answer_index]
    
    # Strategy 2: Look for patterns like "A)", "B)", "C)" in the final part of the response
    # Focus on the last 200 characters to avoid picking up the question options
    response_end = response[-200:] if len(response) > 200 else response
    answer_pattern = r'([A-Z]\))\s*'
    matches = re.findall(answer_pattern, response_end)
    
    if matches:
        # Take the last match (final answer)
        answer_letter = matches[-1][0]  # Take only the letter part, e.g., 'A' from 'A)'
        answer_index = ord(answer_letter) - ord('A')
        if 0 <= answer_index < len(options):
            return options[answer_index]

    # Strategy 2b: Bare letter near the end, e.g. "The best answer is C" or just "C."
    bare_letter = re.findall(r'\b([A-E])\b(?=[^A-E]*$)', response_end, flags=re.IGNORECASE)
    if bare_letter:
        answer_letter = bare_letter[-1].upper()
        answer_index = ord(answer_letter) - ord("A")
        if 0 <= answer_index < len(options):
            return options[answer_index]
    
    # Strategy 3: Look for the model explicitly stating the answer text
    # Look for patterns like "The correct answer is [option text]" or "The answer is [option text]"
    for option in options:
        if isinstance(option, str):
            option_lower = option.lower()
            # Look for explicit statements with the option text
            explicit_text_patterns = [
                f'the correct answer is {re.escape(option_lower)}',
                f'the answer is {re.escape(option_lower)}',
                f'answer: {re.escape(option_lower)}',
                f'final answer: {re.escape(option_lower)}',
                f'best answer is {re.escape(option_lower)}',
                f'correct answer: {re.escape(option_lower)}',
                f'my answer is {re.escape(option_lower)}',
                f'chosen answer: {re.escape(option_lower)}',
                f'selected answer: {re.escape(option_lower)}',
            ]
            
            for pattern in explicit_text_patterns:
                if re.search(pattern, response, flags=re.IGNORECASE):
                    return option
    
    # Strategy 4: Fallback - look for the last occurrence of each option in the response
    # But only in the latter half of the response to avoid the question options
    response_half = response[len(response)//2:]
    response_half_lower = response_half.lower()
    last_occurrence_pos = -1
    last_occurrence_option = None
    
    for option in options:
        if isinstance(option, str):
            option_lower = option.lower()
            pos = response_half_lower.rfind(option_lower)
            if pos > last_occurrence_pos:
                last_occurrence_pos = pos
                last_occurrence_option = option
    
    return last_occurrence_option

def is_multiple_choice_dataset(item: Dict) -> bool:
    """Check if the dataset item is multiple choice or free-form."""
    # Multiple choice datasets have "options" field with string options
    # Free-form datasets have "answers" field instead
    return "options" in item and isinstance(item["options"], list) and len(item["options"]) > 0 and isinstance(item["options"][0], str)

def get_correct_answers(item: Dict, dataset_name: str = None) -> Tuple[List[str], str, bool]:
    """Extract correct answers and determine if multiple choice."""
    is_multiple_choice = is_multiple_choice_dataset(item)
    
    if is_multiple_choice:
        # For winogrande and ethics datasets (multiple choice)
        options = item["options"]
        correct_answer = item["answer"]  # "1", "2", "0", "1", etc.
        
        correct_answer_text = None
        # Convert common encodings to option text:
        # - digits: either 0-indexed or 1-indexed
        # - letters: "A"/"B"/"C"/...
        if isinstance(correct_answer, int):
            idx = correct_answer
            if 0 <= idx < len(options):
                correct_answer_text = options[idx]
            elif 1 <= idx <= len(options):
                correct_answer_text = options[idx - 1]
        elif isinstance(correct_answer, str):
            ca = correct_answer.strip()
            if ca.isdigit():
                idx = int(ca)
                if 0 <= idx < len(options):
                    correct_answer_text = options[idx]
                elif 1 <= idx <= len(options):
                    correct_answer_text = options[idx - 1]
            else:
                m = re.match(r"^\s*([A-E])\s*[\)\.\:\-]?\s*$", ca, flags=re.IGNORECASE)
                if m:
                    letter = m.group(1).upper()
                    idx = ord(letter) - ord("A")
                    if 0 <= idx < len(options):
                        correct_answer_text = options[idx]
                else:
                    # Fall back to literal string (some datasets may store the exact option text)
                    correct_answer_text = ca
        if correct_answer_text is None:
            # Last resort: treat as literal.
            correct_answer_text = str(correct_answer)
        
        return [correct_answer_text], correct_answer_text, True
    else:
        # For complex_web_questions dataset (free-form)
        # The correct answers are stored in item["answers"]["answer"]
        if "answers" in item and "answer" in item["answers"]:
            correct_answers = item["answers"]["answer"]
        else:
            # Fallback to options if answers structure is not available
            correct_answers = item.get("options", [])
        
        primary_answer = correct_answers[0] if correct_answers else ""
        return correct_answers, primary_answer, False

def evaluate_answer(response: str, correct_answers: List[str], is_multiple_choice: bool, options: List[str] = None) -> Tuple[str, bool]:
    """Evaluate if the response is correct."""
    if is_multiple_choice:
        predicted_answer = extract_answer_from_response(response, options)
        is_correct = predicted_answer in correct_answers if predicted_answer else False
    else:
        # For free-form questions, check if any correct answer appears in response
        predicted_answer = response.strip()
        # Handle case where correct_answers might contain lists
        is_correct = False
        for correct_answer in correct_answers:
            if isinstance(correct_answer, list):
                # If correct_answer is a list, check each item
                for item in correct_answer:
                    if isinstance(item, str) and item.lower() in predicted_answer.lower():
                        is_correct = True
                        break
            elif isinstance(correct_answer, str) and correct_answer.lower() in predicted_answer.lower():
                is_correct = True
                break
    
    return predicted_answer, is_correct

def load_results(file_path: str) -> Dict:
    """Load results from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_results(results: Dict, output_file: str):
    """Save results to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def find_result_files(out_dir: str = "out", prefix: Optional[str] = None) -> List[str]:
    """Find result files in the output directory.

    Args:
        out_dir: Directory to search (default: "out")
        prefix: If provided, only include files whose basename starts with this prefix
                (e.g. "exp1_results_").
    """
    result_files: List[str] = []
    out_path = Path(out_dir)

    if not out_path.exists():
        return result_files

    for file_path in out_path.rglob("*.json"):
        name = file_path.name
        if "results" not in name:
            continue
        if prefix is not None and not name.startswith(prefix):
            continue
        result_files.append(str(file_path))

    return sorted(result_files)


def infer_model_from_result_path(file_path: str, out_dir: str = "out") -> str:
    """Infer model key from a results file path.

    Expected new layout:
      out/<model_key>/exp*_results_*.json

    Legacy layout:
      out/exp*_results_*.json  -> returns "combined"
    """
    try:
        out_path = Path(out_dir).resolve()
        rel = Path(file_path).resolve().relative_to(out_path)
        parts = rel.parts
        if len(parts) >= 2:
            top = str(parts[0]).strip()
            if top and top.lower() not in {"figures"}:
                return top
    except Exception:
        pass
    return "combined"

def extract_dataset_name(filename: str) -> str:
    """Extract dataset name from a results filename.

    Expected pattern:
      exp{N}_results_{dataset}_{model}.json

    Notes:
      - Many model identifiers include dots (e.g. Llama-3.3), so we must not stop at the
        first '.' when parsing.
      - We try to split the dataset portion from the model portion using common model
        prefixes as anchors (e.g. "_meta-").
      - If we can't confidently split, we fall back to returning everything after the
        "expN_results_" prefix (without the .json suffix).
    """
    import re
    from pathlib import Path

    name = Path(filename).name
    name = re.sub(r"\.json$", "", name, flags=re.IGNORECASE)

    m = re.match(r"exp\d+[a-z]?_results_(.+)$", name)
    if not m:
        return "unknown"

    remainder = m.group(1)

    # Common anchors that appear right before the model identifier in this repo.
    # Example: exp1_results_winogrande_meta-llama_Llama-3.3-70B-Instruct.json
    anchors = [
        "_meta-",
        "_zai-",
        "_zai_org_",
        "_zai-org_",
        "_openai_",
        "_openai-",
        "_anthropic-",
        "_gpt-",
        "_claude-",
        "_mistral-",
        "_gemini-",
    ]
    for anchor in anchors:
        idx = remainder.find(anchor)
        if idx > 0:
            return remainder[:idx]

    return remainder
