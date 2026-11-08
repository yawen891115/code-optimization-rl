import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from .data import convert_csv, dataset_digest, load_examples, split_by_problem, write_json
from .evaluation import compare_reports, evaluate
from .execution import RunConfig, Runner, discover_tests


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return result


def add_execution(parser):
    parser.add_argument(
        "--tests",
        action="append",
        required=True,
        help="Root containing problem_id/input.*.txt; repeat for public/hidden tests",
    )
    parser.add_argument(
        "--allow-local-execution",
        action="store_true",
        help="Run trusted programs directly on this machine instead of Docker",
    )
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--image", default="python:3.11-slim")


def add_model(parser):
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--adapter")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto"
    )
    parser.add_argument("--max-length", type=positive_int, default=2048)
    parser.add_argument("--seed", type=int, default=42)


def parser_for_cli():
    parser = argparse.ArgumentParser(
        prog="codeopt", description="Execution-guided code optimization"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Convert PIE-style pairs and split by problem ID")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--delimiter", choices=["tab", "comma"], default="tab")
    prepare.add_argument("--limit", type=positive_int)
    prepare.add_argument("--test-fraction", type=float, default=0.2)
    prepare.add_argument("--seed", type=int, default=42)
    tokens = commands.add_parser(
        "tokens", help="Measure and filter complete prompt/response lengths"
    )
    tokens.add_argument("--input", required=True)
    tokens.add_argument("--output", required=True)
    add_model(tokens)
    validate = commands.add_parser(
        "validate", help="Keep pairs whose input and target both pass tests"
    )
    validate.add_argument("--input", required=True)
    validate.add_argument("--output", required=True)
    add_execution(validate)
    for name in ("sft", "rl"):
        command = commands.add_parser(
            name, help="Train LoRA adapters" if name == "sft" else "Train with execution rewards"
        )
        command.add_argument("--input", required=True)
        command.add_argument("--output-dir", required=True)
        add_model(command)
        command.add_argument("--epochs", type=positive_int, default=1)
        command.add_argument("--learning-rate", type=float, default=2e-5)
        command.add_argument("--accumulation", type=positive_int, default=8)
        command.add_argument("--rank", type=positive_int, default=16)
        command.add_argument("--alpha", type=positive_int, default=32)
        command.add_argument("--dropout", type=float, default=0.05)
        if name == "rl":
            command.add_argument("--candidates", type=positive_int, default=2)
            command.add_argument("--max-new-tokens", type=positive_int, default=256)
            command.add_argument("--ranking-weight", type=float, default=1.0)
            command.add_argument("--minimum-speedup", type=float, default=1.05)
            add_execution(command)
    generate = commands.add_parser("generate", help="Generate deterministic candidate programs")
    generate.add_argument("--input", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--max-new-tokens", type=positive_int, default=256)
    generate.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow generation on adapter training problems for debugging",
    )
    add_model(generate)
    evaluate_parser = commands.add_parser(
        "evaluate", help="Evaluate saved candidates against the same tests as the baseline"
    )
    evaluate_parser.add_argument("--input", required=True)
    evaluate_parser.add_argument("--predictions", required=True)
    evaluate_parser.add_argument("--output", required=True)
    add_execution(evaluate_parser)
    compare = commands.add_parser(
        "compare", help="Compare reports with matching datasets and settings"
    )
    compare.add_argument("--run", action="append", required=True, metavar="LABEL=REPORT_JSON")
    compare.add_argument("--output", required=True)
    demo = commands.add_parser(
        "demo", help="Run the bundled handwritten examples without model downloads"
    )
    demo.add_argument("--output-dir", default="runs/demo")
    return parser


def make_runner(args):
    return Runner(
        RunConfig(
            backend="local" if args.allow_local_execution else "docker",
            timeout=args.timeout,
            repeats=args.repeats,
            warmups=args.warmups,
            image=args.image,
        )
    )


def make_model(args):
    from .modeling import ModelConfig

    return ModelConfig(
        model=args.model,
        adapter=args.adapter,
        device=args.device,
        dtype=args.dtype,
        rank=getattr(args, "rank", 16),
        alpha=getattr(args, "alpha", 32),
        dropout=getattr(args, "dropout", 0.05),
    )


def dispatch(args):
    if args.command == "prepare":
        if Path(args.input).suffix in {".json", ".jsonl"}:
            examples = load_examples(args.input)
            if args.limit:
                examples = examples[: args.limit]
            counts = {"kept": len(examples)}
        else:
            examples, counts = convert_csv(
                args.input, "\t" if args.delimiter == "tab" else ",", args.limit
            )
        train, test = split_by_problem(examples, args.test_fraction, args.seed)
        root = Path(args.output_dir)
        write_json(root / "train.json", [e.to_dict() for e in train])
        write_json(root / "test.json", [e.to_dict() for e in test])
        manifest = {
            "seed": args.seed,
            "test_fraction_of_problems": args.test_fraction,
            "conversion": counts,
            "dataset_sha256": dataset_digest(examples),
            "train_samples": len(train),
            "test_samples": len(test),
            "train_problems": sorted({e.problem_id for e in train}),
            "test_problems": sorted({e.problem_id for e in test}),
        }
        write_json(root / "manifest.json", manifest)
        return manifest
    if args.command == "demo":
        from .demo import run_demo

        report = run_demo(args.output_dir)
        return {
            "experiment": report["experiment"],
            "metrics": report["metrics"],
            "report": str(Path(args.output_dir) / "evaluation.json"),
        }
    if args.command == "compare":
        reports = {}
        for entry in args.run:
            label, path = entry.split("=", 1)
            if not label or label in reports:
                raise ValueError("Comparison labels must be nonempty and unique")
            reports[label] = json.loads(Path(path).read_text())
        summary = compare_reports(reports)
        write_json(args.output, summary)
        return summary
    examples = load_examples(args.input)
    if args.command == "tokens":
        from .modeling import require_ml, training_tokens

        _, transformers, _ = require_ml()
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
        lengths, kept = [], []
        for example in examples:
            prefix, target = training_tokens(tokenizer, example, 10**12)
            length = len(prefix) + len(target)
            lengths.append(length)
            if length <= args.max_length:
                kept.append(example.to_dict())
        write_json(args.output, kept)
        return {
            "samples": len(examples),
            "kept": len(kept),
            "max_tokens": max(lengths),
            "mean_tokens": statistics.mean(lengths),
        }
    if args.command == "validate":
        runner = make_runner(args)
        kept, rejected = [], []
        for example in examples:
            cases = discover_tests(args.tests, example.problem_id)
            before, after = runner.run(example.input, cases), runner.run(example.output, cases)
            if before.accepted and after.accepted:
                kept.append(example.to_dict())
            else:
                rejected.append(
                    {
                        "sample_id": example.sample_id,
                        "input": asdict(before),
                        "output": asdict(after),
                    }
                )
        write_json(args.output, kept)
        write_json(str(args.output) + ".rejections.json", rejected)
        return {"kept": len(kept), "rejected": len(rejected)}
    if args.command in {"sft", "rl"}:
        from .rewards import RewardConfig
        from .training import TrainConfig, train

        config = TrainConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            accumulation=args.accumulation,
            max_length=args.max_length,
            max_new_tokens=getattr(args, "max_new_tokens", 256),
            candidates=getattr(args, "candidates", 2),
            ranking_weight=getattr(args, "ranking_weight", 1.0),
            seed=args.seed,
        )
        return train(
            examples,
            make_model(args),
            config,
            args.output_dir,
            mode=args.command,
            runner=make_runner(args) if args.command == "rl" else None,
            test_roots=getattr(args, "tests", []),
            reward_config=RewardConfig(minimum_speedup=getattr(args, "minimum_speedup", 1.05)),
        )
    if args.command == "generate":
        from .training import generate_predictions

        predictions = generate_predictions(
            examples,
            make_model(args),
            args.max_length,
            args.max_new_tokens,
            args.seed,
            args.allow_overlap,
        )
        write_json(args.output, predictions)
        write_json(
            str(args.output) + ".manifest.json",
            {
                "model": asdict(make_model(args)),
                "dataset_sha256": dataset_digest(examples),
                "seed": args.seed,
                "decoding": "greedy",
                "max_new_tokens": args.max_new_tokens,
            },
        )
        return {"generated": len(predictions), "output": args.output}
    rows = json.loads(Path(args.predictions).read_text())
    predictions = {row["sample_id"]: row["generated"] for row in rows}
    if len(rows) != len(predictions):
        raise ValueError("Duplicate prediction IDs are not allowed")
    report = evaluate(examples, predictions, make_runner(args), args.tests)
    write_json(args.output, report)
    return report["metrics"]


def main(argv=None):
    parser = parser_for_cli()
    args = parser.parse_args(argv)
    try:
        result = dispatch(args)
    except (ValueError, TypeError, RuntimeError, OSError, KeyError) as error:
        parser.exit(2, f"codeopt: {error}\n")
    print(json.dumps(result, indent=2))
