# Validation status

Validated on 2026-11-08 with Python 3.12.14 and CPU PyTorch 2.8.0. Direct dependency versions are recorded in `requirements-tested.txt`.

| Check | Result |
| --- | --- |
| Automated regression suite | **38 passed** |
| Ruff static checks | Passed |
| LoRA gradient flow | Only adapter parameters changed after an optimizer step |
| Response loss | Matched manually selected next-token log probabilities; prompt tokens excluded |
| Partial accumulation | Correct average for an incomplete final batch |
| Tiny-model instruction tuning | Adapter saved and reloaded successfully |
| Tiny-model generation | Held-out prompt produced a candidate after adapter reload |
| Reward-training integration | Controlled correct/incorrect candidates executed, ranked, backpropagated, and checkpointed |
| Data leakage checks | Grouped splits and inherited adapter training IDs enforced |
| Original project data compatibility | 1,192 distinct examples loaded; 875 train / 317 test samples across disjoint problem groups |
| Handwritten demo | Both example programs passed their supplied tests |

The tiny model is constructed locally with random weights. Reward-training integration uses controlled candidate text to make the correctness/reward assertions deterministic. These checks establish software behavior; they are not evidence of improvements from a pretrained LLM.

The local execution backend was tested. Docker is not available in the validation environment, so the Docker execution path remains unverified here. A full Llama 3 8B training run, full PIE/CodeNet benchmark, GPU memory profiling, and statistical runtime comparisons have not been run. No pretrained weights or inherited benchmark result files are included.

To reproduce:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-tested.txt
python -m pip install -e .
ruff check src tests
pytest -q
codeopt demo --output-dir runs/demo
```
