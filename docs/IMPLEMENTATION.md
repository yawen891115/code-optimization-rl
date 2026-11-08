# Implementation notes

## Package structure

| Module | Responsibility |
| --- | --- |
| `data.py` | Schema validation, sample hashes, CSV conversion, and grouped splits |
| `modeling.py` | Chat templates, LoRA adapters, generation, and response likelihood |
| `training.py` | Instruction tuning, reward ranking, accumulation, and checkpoints |
| `execution.py` / `_worker.py` | Test discovery, container isolation, timeouts, and measurements |
| `rewards.py` | Correctness-gated reward policy |
| `evaluation.py` | Comparable records and aggregate metrics |
| `cli.py` | Command-line entry points |

## Training objective

Instruction tuning uses negative mean log probability of the optimized response conditioned on the full prompt. Prompt labels are masked. The forward pass uses a causal language model; gradients flow through every scored response token.

For reward training, each input produces at least two sampled candidates. The complete candidate text is scored by the model while its code-only form is executed. Rewards are:

| Outcome | Reward |
| --- | ---: |
| Python syntax error | -1.0 |
| Wrong answer, runtime error, or timeout | 0.0 |
| Correct without a sufficient speedup | 0.5 |
| Correct with at least 1.05× speedup | 1.0 |

The original program must pass the same nonempty test suite before its timing is used as a reference. `--minimum-speedup` controls the noise margin.

Let `s_i` be the mean conditional log probability of candidate `i` and `r_i` its reward. Each pair where `r_i > r_j` contributes `(r_i - r_j) * softplus(s_j - s_i)` to the ranking loss. The average ranking loss is added to the negative score of the highest-reward candidate when that candidate is correct. Equal nonpositive rewards skip the update. This is execution-reward-ranked fine-tuning with a best-candidate likelihood term.

Only LoRA parameters are optimized. Losses stay as tensors until after backpropagation. Gradient accumulation is normalized correctly for a partial final batch, and gradients are clipped before each optimizer step. Checkpoints contain adapters and tokenizer metadata; they are inference/reinitialization checkpoints, not exact optimizer-state resumes.

## Measurement contract

The runner executes tests sequentially to reduce contention. Each test has configurable warmup and measured repetitions. A sample's runtime is the arithmetic mean of per-test median wall times. Measurements include child interpreter startup and exclude container startup. They do not measure peak memory or algorithm-only CPU time.

Outputs are compared exactly after stripping leading/trailing whitespace. Numeric tolerances and custom judges are not implemented. Programs run with the standard library provided by the selected Python image. Missing test files fail explicitly.

Correctness percentage is `accepted / all samples`. For samples where both programs pass, speedup is `original_time / generated_time`, aggregated geometrically. Mean runtime reduction is `100 * (1 - generated_time / original_time)` and includes negative values for slowdowns. Code length counts nonblank lines, including comment lines.

Comparisons require matching dataset hashes, test-content hashes, and execution settings. They should also be collected on the same idle machine; these checks do not establish identical CPU load or thermal conditions.

## Changes introduced in this version

- A packaged CLI replaces scripts with hard-coded filesystem paths.
- LoRA instruction tuning uses a causal-LM objective compatible with Llama.
- Reward training keeps differentiable scores and uses multiple candidates.
- Dataset and adapter manifests make accidental train/test overlap visible.
- Runtime metrics include regressions and reject missing test suites.
- Repeated measurements, resource limits, and a container runner make evaluation more controlled.
- Synthetic tiny-model tests check gradient flow and adapter serialization without downloading pretrained weights.
