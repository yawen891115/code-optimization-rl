"""LoRA instruction tuning and execution-reward-ranked fine-tuning."""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import dataset_digest, write_json
from .execution import discover_tests
from .modeling import (
    generate_candidates,
    load_model,
    parameter_counts,
    prompt_tokens,
    require_ml,
    response_logprob,
    training_tokens,
)
from .rewards import RewardConfig, execution_reward


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 1
    learning_rate: float = 2e-5
    accumulation: int = 8
    max_length: int = 2048
    max_new_tokens: int = 256
    candidates: int = 2
    ranking_weight: float = 1.0
    seed: int = 42

    def __post_init__(self):
        if self.epochs < 1 or self.accumulation < 1 or self.learning_rate <= 0:
            raise ValueError("Training counts and learning rate must be positive")
        if self.max_length < 2 or self.max_new_tokens < 1 or self.candidates < 2:
            raise ValueError("RL needs at least two candidates and a positive token budget")
        if self.ranking_weight < 0:
            raise ValueError("ranking_weight cannot be negative")


def ranked_objective(scores, rewards, ranking_weight=1.0):
    import torch
    from torch.nn import functional as F

    if len(scores) != len(rewards) or len(scores) < 2:
        raise ValueError("At least two paired scores and rewards are required")
    comparisons = [
        F.softplus(scores[j] - scores[i]) * (rewards[i] - rewards[j])
        for i in range(len(scores))
        for j in range(len(scores))
        if rewards[i] > rewards[j]
    ]
    zero = scores.sum() * 0
    ranking = torch.stack(comparisons).mean() if comparisons else zero
    best = max(range(len(rewards)), key=rewards.__getitem__)
    tuning = -scores[best] if rewards[best] > 0 else zero
    return tuning + ranking_weight * ranking, ranking, tuning


class GradientAccumulator:
    def __init__(self, model, optimizer, steps):
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.optimizer, self.steps, self.pending = optimizer, steps, 0
        self.updates = 0
        self.optimizer.zero_grad(set_to_none=True)

    def backward(self, loss):
        import torch

        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        (loss / self.steps).backward()
        self.pending += 1
        if self.pending == self.steps:
            self.flush()

    def flush(self):
        import torch

        if not self.pending:
            return
        if self.pending < self.steps:
            for parameter in self.parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(self.steps / self.pending)
        torch.nn.utils.clip_grad_norm_(self.parameters, max_norm=1.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        self.optimizer.zero_grad(set_to_none=True)
        self.pending = 0


def train(
    examples,
    model_config,
    config,
    output_dir,
    mode="sft",
    runner=None,
    test_roots=(),
    reward_config=None,
):
    if mode not in {"sft", "rl"}:
        raise ValueError("mode must be sft or rl")
    if not examples:
        raise ValueError("Training dataset is empty")
    if mode == "rl" and runner is None:
        raise ValueError("RL training requires an execution runner")
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Output directory must be empty to avoid overwriting a run")
    reward_config = reward_config or RewardConfig()
    torch, transformers, _ = require_ml()
    transformers.set_seed(config.seed)
    model, tokenizer = load_model(model_config, trainable=True)
    model.train()
    encoded = [
        (example, training_tokens(tokenizer, example, config.max_length)) for example in examples
    ]
    usable = [(example, tokens) for example, tokens in encoded if tokens is not None]
    if not usable:
        raise ValueError("All examples exceed the token budget; no code is silently truncated")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config.learning_rate
    )
    gradients = GradientAccumulator(model, optimizer, config.accumulation)
    test_map, baseline_cache = {}, {}
    if mode == "rl":
        test_map = {e.problem_id: discover_tests(test_roots, e.problem_id) for e, _ in usable}
        if any(not cases for cases in test_map.values()):
            raise ValueError("Every RL training problem requires a nonempty test suite")
    root.mkdir(parents=True, exist_ok=True)
    training_problems = {e.problem_id for e in examples}
    if model_config.adapter:
        parent_manifest = Path(model_config.adapter) / "training_manifest.json"
        if parent_manifest.is_file():
            training_problems.update(
                json.loads(parent_manifest.read_text())["training_problem_ids"]
            )
    manifest = {
        "mode": mode,
        "model": asdict(model_config),
        "training": asdict(config),
        "dataset_sha256": dataset_digest(examples),
        "training_problem_ids": sorted(training_problems),
        "usable_samples": len(usable),
        "overlength_samples": len(examples) - len(usable),
        "parameters": parameter_counts(model),
        "updates": 0,
        "objective": "response_nll"
        if mode == "sft"
        else "pairwise_reward_ranking_plus_best_correct_nll",
    }
    if mode == "rl":
        manifest.update(reward=asdict(reward_config), execution=runner.fingerprint())
    write_json(root / "run.json", manifest)
    rng = random.Random(config.seed)
    update_count = 0
    with (root / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(config.epochs):
            order = list(usable)
            rng.shuffle(order)
            for example, (prefix, target) in order:
                record = {"epoch": epoch + 1, "sample_id": example.sample_id}
                if mode == "sft":
                    loss = -response_logprob(model, prefix, target)
                else:
                    if len(prefix) + config.max_new_tokens > config.max_length:
                        log.write(
                            json.dumps({**record, "skipped": "generation_token_budget"}) + "\n"
                        )
                        continue
                    baseline = baseline_cache.get(example.sample_id)
                    if baseline is None:
                        baseline = runner.run(example.input, test_map[example.problem_id])
                        baseline_cache[example.sample_id] = baseline
                    if not baseline.accepted:
                        log.write(
                            json.dumps(
                                {
                                    **record,
                                    "skipped": "invalid_baseline",
                                    "verdict": baseline.verdict,
                                }
                            )
                            + "\n"
                        )
                        continue
                    candidates = generate_candidates(
                        model,
                        tokenizer,
                        prefix,
                        count=config.candidates,
                        max_new_tokens=config.max_new_tokens,
                        temperature=0.8,
                    )
                    executions = [
                        runner.run(c["generated"], test_map[example.problem_id]) for c in candidates
                    ]
                    rewards = [
                        execution_reward(result, baseline, reward_config) for result in executions
                    ]
                    if max(rewards) <= 0 and len(set(rewards)) == 1:
                        log.write(
                            json.dumps(
                                {
                                    **record,
                                    "skipped": "equal_nonpositive_rewards",
                                    "rewards": rewards,
                                }
                            )
                            + "\n"
                        )
                        continue
                    scores = torch.stack(
                        [response_logprob(model, prefix, c["tokens"]) for c in candidates]
                    )
                    loss, ranking, tuning = ranked_objective(scores, rewards, config.ranking_weight)
                    record.update(
                        rewards=rewards,
                        ranking_loss=ranking.detach().item(),
                        tuning_loss=tuning.detach().item(),
                        verdicts=[result.verdict for result in executions],
                    )
                gradients.backward(loss)
                update_count += 1
                log.write(json.dumps({**record, "loss": loss.detach().item()}) + "\n")
                log.flush()
            gradients.flush()
            checkpoint = root / f"epoch-{epoch + 1}"
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            write_json(
                checkpoint / "training_manifest.json",
                {
                    **manifest,
                    "updates": gradients.updates,
                    "examples_optimized": update_count,
                },
            )
    if not update_count:
        raise RuntimeError(f"No training updates were made; inspect {root / 'metrics.jsonl'}")
    final = root / "adapter"
    model.save_pretrained(final, safe_serialization=True)
    tokenizer.save_pretrained(final)
    manifest["updates"] = gradients.updates
    manifest["examples_optimized"] = update_count
    write_json(final / "training_manifest.json", manifest)
    write_json(root / "run.json", manifest)
    return manifest


def generate_predictions(
    examples, model_config, max_length=2048, max_new_tokens=256, seed=42, allow_overlap=False
):
    if model_config.adapter:
        path = Path(model_config.adapter) / "training_manifest.json"
        if path.is_file() and not allow_overlap:
            training_problems = set(json.loads(path.read_text())["training_problem_ids"])
            if training_problems & {e.problem_id for e in examples}:
                raise ValueError("Evaluation problems overlap adapter training problems")
    _, transformers, _ = require_ml()
    transformers.set_seed(seed)
    model, tokenizer = load_model(model_config)
    model.eval()
    predictions = []
    for example in examples:
        prefix = prompt_tokens(tokenizer, example)
        if len(prefix) + max_new_tokens > max_length:
            raise ValueError(f"{example.sample_id}: prompt plus response budget exceeds max_length")
        candidate = generate_candidates(
            model, tokenizer, prefix, max_new_tokens=max_new_tokens, temperature=0
        )[0]
        predictions.append({"sample_id": example.sample_id, "generated": candidate["generated"]})
    return predictions
