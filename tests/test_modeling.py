import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from peft import LoraConfig, TaskType, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from codeopt.data import Example
from codeopt.execution import RunConfig, Runner
from codeopt.modeling import ModelConfig, load_model, response_logprob, training_tokens
from codeopt.training import TrainConfig, generate_predictions, ranked_objective, train


def test_partial_accumulation_averages_only_actual_examples():
    from codeopt.training import GradientAccumulator

    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    accumulator = GradientAccumulator(model, optimizer, steps=3)
    accumulator.backward(model.weight.sum() * 0.1)
    accumulator.backward(model.weight.sum() * 0.3)
    accumulator.flush()
    assert model.weight.item() == pytest.approx(-0.02)
    assert accumulator.updates == 1


@pytest.fixture
def tiny_model_dir(tmp_path):
    torch.set_num_threads(1)
    vocabulary = {
        word: index
        for index, word in enumerate(
            [
                "[UNK]",
                "[EOS]",
                "[PAD]",
                "print",
                "(",
                ")",
                "1",
                "2",
                "3",
                "input",
                "+",
                "int",
                "###",
                "Response",
                "Instruction",
            ]
        )
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]"
    )
    config = LlamaConfig(
        vocab_size=len(vocabulary),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        eos_token_id=1,
        pad_token_id=2,
    )
    model = LlamaForCausalLM(config)
    directory = tmp_path / "base"
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return directory


def test_response_loss_preserves_gradients_and_only_updates_adapters(tiny_model_dir):
    torch.manual_seed(7)
    model = LlamaForCausalLM.from_pretrained(tiny_model_dir)
    model = get_peft_model(
        model,
        LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4, target_modules="all-linear"),
    )
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    scores = torch.stack(
        [response_logprob(model, [3, 4], target) for target in ([6, 5, 1], [7, 5, 1])]
    )
    loss, _, _ = ranked_objective(scores, [1.0, 0.0])
    assert loss.requires_grad
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad
    )
    torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1).step()
    changed = [name for name, p in model.named_parameters() if not torch.equal(before[name], p)]
    assert changed and all("lora_" in name for name in changed)


def test_prompt_tokens_do_not_contribute_to_response_loss(tiny_model_dir):
    model = LlamaForCausalLM.from_pretrained(tiny_model_dir)
    model.eval()
    prefix, target = [3, 4, 6], [5, 1]
    ids = torch.tensor([prefix + target])
    logits = model(ids).logits
    expected = torch.log_softmax(logits.float(), dim=-1)[0, [2, 3], [5, 1]].mean()
    assert torch.allclose(response_logprob(model, prefix, target), expected, atol=1e-6)


def test_overlength_code_is_dropped_without_truncation(tiny_model_dir):
    _, tokenizer = load_model(ModelConfig(model=str(tiny_model_dir)))
    example = Example("a", "print(1)", "print(1)")
    assert training_tokens(tokenizer, example, 3) is None
    prefix, target = training_tokens(tokenizer, example, 256)
    assert len(prefix) > 3 and target[-1] == tokenizer.eos_token_id


def test_equal_failed_candidates_do_not_get_positive_tuning_loss():
    scores = torch.tensor([-2.0, -3.0], requires_grad=True)
    loss, ranking, tuning = ranked_objective(scores, [0.0, 0.0])
    assert loss.item() == ranking.item() == tuning.item() == 0


def test_lora_save_load_and_training_overlap_guard(tiny_model_dir, tmp_path):
    examples = [Example("train", "print(1)", "print(1)")]
    output = tmp_path / "sft"
    manifest = train(
        examples,
        ModelConfig(model=str(tiny_model_dir), rank=2, alpha=4),
        TrainConfig(accumulation=2, max_length=256),
        output,
    )
    assert manifest["updates"] == 1
    assert 0 < manifest["parameters"]["trainable_percent"] < 100
    assert (output / "adapter" / "adapter_model.safetensors").is_file()
    configuration = ModelConfig(model=str(tiny_model_dir), adapter=str(output / "adapter"))
    with pytest.raises(ValueError, match="overlap"):
        generate_predictions(examples, configuration)
    held_out = [Example("test", "print(2)", "print(2)")]
    predictions = generate_predictions(held_out, configuration, max_length=256, max_new_tokens=4)
    assert len(predictions) == 1 and predictions[0]["sample_id"] == held_out[0].sample_id


def test_rl_training_executes_candidates_and_saves_adapter(tiny_model_dir, tmp_path, monkeypatch):
    import codeopt.training as training

    def candidates(model, tokenizer, prefix, **kwargs):
        programs = ["print(1)", "print(2)"]
        return [
            {
                "generated": text,
                "tokens": tokenizer.encode(text, add_special_tokens=False)
                + [tokenizer.eos_token_id],
            }
            for text in programs
        ]

    monkeypatch.setattr(training, "generate_candidates", candidates)
    root = tmp_path / "tests" / "p1"
    root.mkdir(parents=True)
    (root / "input.0.txt").write_text("")
    (root / "output.0.txt").write_text("1\n")
    output = tmp_path / "rl"
    report = train(
        [Example("p1", "print(1)", "print(1)")],
        ModelConfig(model=str(tiny_model_dir), rank=2, alpha=4),
        TrainConfig(max_length=256, max_new_tokens=16, accumulation=1),
        output,
        mode="rl",
        runner=Runner(RunConfig(backend="local", repeats=1, warmups=0)),
        test_roots=[root.parent],
    )
    record = json.loads((output / "metrics.jsonl").read_text().splitlines()[0])
    assert report["updates"] == 1
    assert record["verdicts"] == ["accepted", "wrong_answer"]
    assert record["rewards"][0] > record["rewards"][1]
    assert (output / "adapter" / "adapter_model.safetensors").is_file()


def test_continued_training_retains_prior_problem_ids(tiny_model_dir, tmp_path):
    first = tmp_path / "first"
    train(
        [Example("earlier", "print(1)", "print(1)")],
        ModelConfig(model=str(tiny_model_dir), rank=2, alpha=4),
        TrainConfig(max_length=256),
        first,
    )
    second = tmp_path / "second"
    manifest = train(
        [Example("later", "print(2)", "print(2)")],
        ModelConfig(model=str(tiny_model_dir), adapter=str(first / "adapter")),
        TrainConfig(max_length=256),
        second,
    )
    assert manifest["training_problem_ids"] == ["earlier", "later"]
    with pytest.raises(ValueError, match="overlap"):
        generate_predictions(
            [Example("earlier", "print(3)", "print(3)")],
            ModelConfig(model=str(tiny_model_dir), adapter=str(second / "adapter")),
        )
