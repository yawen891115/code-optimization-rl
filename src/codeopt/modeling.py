"""Shared causal-LM prompting, LoRA loading and differentiable sequence scores."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    adapter: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05


def require_ml():
    try:
        import peft
        import torch
        import transformers
    except ImportError as error:
        raise RuntimeError('Install the model dependencies with pip install -e ".[ml]"') from error
    return torch, transformers, peft


def load_model(config, trainable=False):
    torch, transformers, peft = require_ml()
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.dtype == "auto":
        dtype = (
            torch.bfloat16
            if device.startswith("cuda") and torch.cuda.is_bf16_supported()
            else torch.float32
        )
    else:
        dtype = getattr(torch, config.dtype)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must provide an EOS or PAD token")
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model,
        torch_dtype=dtype,
        trust_remote_code=False,
    ).to(device)
    if config.adapter:
        model = peft.PeftModel.from_pretrained(model, config.adapter, is_trainable=trainable)
    elif trainable:
        if config.rank < 1 or config.alpha < 1 or not 0 <= config.dropout < 1:
            raise ValueError("Invalid LoRA configuration")
        model = peft.get_peft_model(
            model,
            peft.LoraConfig(
                task_type=peft.TaskType.CAUSAL_LM,
                r=config.rank,
                lora_alpha=config.alpha,
                lora_dropout=config.dropout,
                target_modules="all-linear",
                bias="none",
            ),
        )
    if trainable:
        model.config.use_cache = False
    return model, tokenizer


def prompt_tokens(tokenizer, example):
    content = f"{example.instruction}\n\n{example.input}"
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
        )
    return tokenizer.encode(
        f"### Instruction\n{content}\n\n### Response\n", add_special_tokens=True
    )


def training_tokens(tokenizer, example, max_length):
    prefix = prompt_tokens(tokenizer, example)
    if tokenizer.chat_template:
        full = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": f"{example.instruction}\n\n{example.input}"},
                {"role": "assistant", "content": example.output},
            ],
            tokenize=True,
            add_generation_prompt=False,
        )
        if full[: len(prefix)] != prefix:
            raise ValueError("Chat template does not have a stable assistant prefix")
        target = full[len(prefix) :]
    else:
        target = tokenizer.encode(example.output, add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            target.append(tokenizer.eos_token_id)
    if not prefix or not target:
        raise ValueError("Empty prompt or response token sequence")
    if len(prefix) + len(target) > max_length:
        return None
    return prefix, target


def response_logprob(model, prefix, target):
    import torch
    from torch.nn import functional as F

    if not prefix or not target:
        raise ValueError("Scoring requires both prompt and target tokens")
    device = next(model.parameters()).device
    ids = torch.tensor([prefix + target], dtype=torch.long, device=device)
    labels = ids.clone()
    labels[:, : len(prefix)] = -100
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    losses = F.cross_entropy(
        logits[:, :-1].float().transpose(1, 2),
        labels[:, 1:],
        ignore_index=-100,
        reduction="none",
    )
    return -losses.sum() / (labels[:, 1:] != -100).sum()


def clean_code(text):
    match = re.fullmatch(
        r"\s*```(?:python|py)?\s*\n(.*?)\n```\s*", text, flags=re.DOTALL | re.IGNORECASE
    )
    return match.group(1).strip() if match else text.strip()


def generate_candidates(model, tokenizer, prefix, count=1, max_new_tokens=256, temperature=0.7):
    import torch

    if count < 1 or max_new_tokens < 1 or temperature < 0:
        raise ValueError("Invalid generation settings")
    if temperature == 0 and count != 1:
        raise ValueError("Greedy decoding supports one candidate")
    device = next(model.parameters()).device
    ids = torch.tensor([prefix], dtype=torch.long, device=device)
    previous_mode = model.training
    model.eval()
    settings = {
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": count,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    if temperature > 0:
        settings.update(temperature=temperature, top_p=0.95)
    try:
        with torch.no_grad():
            outputs = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **settings)
    finally:
        model.train(previous_mode)
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    candidates = []
    for sequence in outputs:
        target = sequence[len(prefix) :].tolist()
        for index, token in enumerate(target):
            if token in eos_ids:
                target = target[: index + 1]
                break
        candidates.append(
            {
                "tokens": target,
                "generated": clean_code(tokenizer.decode(target, skip_special_tokens=True)),
            }
        )
    return candidates


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_percent": 100 * trainable / total}
