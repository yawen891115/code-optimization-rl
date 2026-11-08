# Code Optimization with RL

**PyTorch · Hugging Face · LoRA · execution rewards**

A Python toolkit for adapting code language models to produce faster programs while preserving correctness. The pipeline combines instruction tuning, execution-reward-ranked training, and matched evaluation of baseline and adapted models.

Maintained by **Yawen Zheng**.

## What is implemented

| Stage | Behavior |
| --- | --- |
| Data | PIE-style CSV/TSV conversion, Python syntax filtering, stable sample IDs, and reproducible splits by problem ID |
| Instruction tuning | Causal-LM LoRA adapters, response-only loss, gradient accumulation, and epoch checkpoints |
| Reward training | Multiple sampled candidates, correctness-gated execution rewards, differentiable pairwise ranking, and best-correct-candidate likelihood |
| Evaluation | Test verdicts, repeated runtime measurements, code length, and comparison of runs using identical data and settings |
| Reproducibility | Dataset/test hashes, run manifests, explicit seeds, adapter provenance checks, and automated tests |

## Quick start

Python 3.10+ is required. The data tools and handwritten demo use only the standard library.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
codeopt demo --output-dir runs/demo
```

The demo writes example pairs, test cases, predictions, and an evaluation report. It checks two handwritten programs; it does **not** represent trained-model performance. Short-program timings include interpreter startup and can fluctuate.

For model training:

```bash
pip install -e '.[ml,dev]'
pytest -q
```

The default model is `meta-llama/Meta-Llama-3-8B-Instruct`. Access to gated weights must be configured through Hugging Face. An appropriate GPU is needed for practical 8B-model training; a complete pretrained-model training run has not been performed for this repository release.

## Prepare a dataset

Obtain the [PIE performance-improvement pairs and CodeNet tests](https://github.com/madaan/pie-perf#dataset). Large datasets and weights are kept outside version control.

```bash
codeopt prepare --input data/raw/pairs.tsv --output-dir data/split --test-fraction 0.2
codeopt tokens --input data/split/train.json --output data/train.json --max-length 2048
codeopt tokens --input data/split/test.json --output data/test.json --max-length 2048
```

Each example contains `problem_id`, `instruction`, `input` (original code), and `output` (optimized reference code). Prepared files also contain a content-derived `sample_id`. All versions of a problem stay in the same split. Code that exceeds the token budget is excluded whole.

Test directories use `<root>/<problem_id>/input.<n>.txt` and matching `output.<n>.txt`. Repeat `--tests` to include public and hidden test roots.

## Train adapters

```bash
codeopt sft --input data/train.json --output-dir runs/sft --device cuda

docker pull python:3.11-slim
codeopt rl --input data/train.json --adapter runs/sft/adapter \
  --output-dir runs/rl --device cuda --tests data/public_tests \
  --tests data/hidden_tests --candidates 2
```

The execution runner uses Docker with networking disabled, read-only mounts, and resource limits. `--allow-local-execution` is available for trusted programs on a disposable machine; local subprocess execution is not a security sandbox. Docker must be installed and its image available before evaluation or reward training.

Each run stores `run.json`, per-example `metrics.jsonl`, epoch adapters, and a final `adapter/`. The manifest records which problems were used for training. Generation rejects overlap with those problem IDs when the adapter was produced by this toolkit.

## Compare baseline and adapted models

```bash
codeopt generate --input data/test.json --output runs/base-predictions.json --device cuda
codeopt generate --input data/test.json --adapter runs/rl/adapter \
  --output runs/rl-predictions.json --device cuda

codeopt evaluate --input data/test.json --predictions runs/base-predictions.json \
  --tests data/public_tests --tests data/hidden_tests --output runs/base-evaluation.json
codeopt evaluate --input data/test.json --predictions runs/rl-predictions.json \
  --tests data/public_tests --tests data/hidden_tests --output runs/rl-evaluation.json

codeopt compare --run baseline=runs/base-evaluation.json \
  --run reward_tuned=runs/rl-evaluation.json --output runs/comparison.json
```

Run the same sequence with `runs/sft/adapter` to add the instruction-tuned checkpoint. Correctness uses all evaluated samples; speedup uses samples where both the original and candidate pass. Slowdowns remain in aggregate metrics. No inherited benchmark scores are presented as results of this implementation.

See [implementation details](docs/IMPLEMENTATION.md), [validation status](docs/VALIDATION.md), and [project background and notices](THIRD_PARTY_NOTICES.md).
