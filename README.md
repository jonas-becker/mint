<div align="center">

  <h1>Misinformation Propagation in Benign Multi-Agent Systems</h1>
  <h3>...and the MINT Dataset</h3>

  <p>
    <a href="https://github.com/Multi-Agent-LLMs/mallm">MALLM Framework</a>
  </p>
</div>

This is the offical repository for the paper "Misinformation Propagation in Benign Multi-Agent Systems".

## What does MINT do?

MINT studies how **misinformation** affects large language models in **single-agent** and **multi-agent debate** settings. The repository provides:

- the **MINT dataset** with task instances, false facts, and eight misinformation strategies
- experiment scripts for single-agent baselines, multi-agent debates (via [MALLM](https://github.com/Multi-Agent-LLMs/mallm)), and agent-composition sweeps
- plotting utilities to reproduce paper figures

## Install

Create an environment:

```bash
conda create --name mint python=3.11
conda activate mint
pip install torch transformers datasets numpy pandas matplotlib seaborn tqdm requests
```

Clone [MALLM](https://github.com/Multi-Agent-LLMs/mallm) next to this repository (required for `exp2.py`, `exp2_ablation.py`, and `exp3.py`):

```text
github/
  mint/          # this repo
  mallm/         # MALLM framework
```

Multi-agent experiments expect an **OpenAI-compatible** model endpoint (vLLM, SGLang, TGI, or similar). Slurm job scripts are provided for cluster runs.

## Dataset

The released benchmark is in `MINT-dataset_v1.1/`:

| Dataset | Task type |
|---|---|
| `winogrande_misinformed.json` | Commonsense reasoning (multiple choice) |
| `ethics_commonsense_misinformed.json` | Moral judgment (multiple choice) |
| `complex_web_questions_misinformed.json` | Complex QA (free-form) |

Each instance includes a `false_fact`, `misinformation_by_strategy` (`clickbait`, `hoax`, `rumor`, `satire`, `propaganda`, `framing`, `conspiracy`, `other`), and `irrelevant_true_information` as a control.

To regenerate or extend datasets:

```bash
python download_datasets.py --use_config_samples --datasets winogrande ethics_commonsense complex_web_questions
```

## Run Experiments

### Exp 1 — Single agent

Baseline vs. misinformed single-agent prompting (local HuggingFace or OpenAI-compatible API):

```bash
python exp1.py --model_name meta-llama/Llama-3.3-70B-Instruct --inference openai --endpoint_url http://127.0.0.1:8080/v1
```

`exp1a.py` runs the same setup with **irrelevant true information** instead of misinformation.

### Exp 2 — Multi-agent debate

3-agent MALLM debates across datasets and misinformation strategies:

```bash
python exp2.py --endpoint_url http://127.0.0.1:8080/v1 --model_name meta-llama/Llama-3.3-70B-Instruct
```

`exp2_ablation.py` runs the same setup **without** misinformation.

### Exp 3 — Agent composition

5-agent debates on WinoGrande, sweeping the number of misinformed agents (0–5):

```bash
python exp3.py --endpoint_url http://127.0.0.1:8080/v1 --model_name meta-llama/Llama-3.3-70B-Instruct
```

Quick smoke test (no model server):

```bash
python exp2.py --mock --debug
```

Results are written to `out/<model_name>/`. Use `--continue` to resume unfinished runs.

## Figures

Generate plots from saved results:

```bash
python exp1_figures.py
python exp2_figures.py
python exp3_figures.py
python exp1_exp2_comparison.py
```

## Code Structure

| Component | Description |
|---|---|
| `download_datasets.py` | Download, sample, and generate misinformed datasets |
| `shared_utils.py` | Prompts, loading, evaluation helpers |
| `exp1.py` / `exp1a.py` | Single-agent experiments |
| `exp2.py` / `exp2_ablation.py` | Multi-agent debate experiments (MALLM) |
| `exp3.py` | Misinformed vs. informed agent composition |
| `exp*_figures.py` | Figure generation |
| `*.slurm` | Cluster job templates (model server + experiment) |
| `MINT-dataset_v1.1/` | Released benchmark data |

## Citation

If you use this repository, please cite the paper and the MALLM framework:

```bibtex
@misc{becker2026,
  author={Becker, Jonas and Wahle, Jan Philip and Ruas, Terry and Gipp, Bela},
  title={Misinformation Propagation in Benign Multi-Agent Systems},
  year={2026},
  month={06}
}
```

```bibtex
@inproceedings{becker-etal-2025-mallm,
    title = "{MALLM}: Multi-Agent Large Language Models Framework",
    author = "Becker, Jonas and Kaesberg, Lars Benedikt and Bauer, Niklas and Wahle, Jan Philip and Ruas, Terry and Gipp, Bela",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    year = "2025",
    url = "https://aclanthology.org/2025.emnlp-demos.29/"
}
```
