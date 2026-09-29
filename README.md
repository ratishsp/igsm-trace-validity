# Correct Answers, Invalid Traces

Code for *Correct Answers, Invalid Traces: What Verifiable Grade-School Math
Reveals About Chain-of-Thought Traces* (Puduppully, Misra, Iyer, Kalwar, Palod
and Kambhampati, 2026, [arXiv 2609.38107](https://arxiv.org/abs/2609.38107)). The paper trains small transformers on iGSM (Ye et al.,
ICLR 2025), where every emitted trace can be checked mechanically, and measures
how often a correct answer comes with a valid trace.

## Setup

```
git clone https://github.com/facebookresearch/iGSM iGSM   # generator, commit a1ed1d0
pip install -r requirements.txt
export IGSM_ROOT=$PWD/iGSM
python -m pytest tests -q
```

## Training

`slurm/train_template.sh` holds the paper's training command (100k steps at batch
512, seed 42). The training trace is selected by flags.

| model | flags |
|---|---|
| clean | none |
| swapped trace | `--swap_traces` |
| op-matched swapped trace | `--swap_traces --swap_match_op` |
| shuffled prefix | `--prefix_noise_frac 0.3` (0.1, 0.5, 0.75, 1.0) |
| no trace | `--no_trace` |
| non-minimal traces | `--redundant_trace` |
| non-minimal traces, 90% | `--redundant_frac 0.9` |

```
sbatch slurm/train_template.sh outputs/clean
sbatch slurm/train_template.sh outputs/swapped --swap_traces
```

## Evaluation

`slurm/eval_template.sh` wraps `evaluate.py`. The paper's test sets are the exact-op
levels 15, 20, 21, 22 and 23 (4096 problems each, seed 4) and the in-distribution
mixture (op <= 15, eight chunks of 512 problems, seeds 4 to 11).

```
sbatch slurm/eval_template.sh outputs/clean/checkpoint-100000 evals/clean --only_op 23 --num_problems 4096
sbatch slurm/eval_template.sh outputs/clean/checkpoint-100000 evals/clean_mix --only_level 'op<=15' --num_problems 512 --seed 4
```

`--reask` re-points the question of each problem at another parameter of the same
world; the wide world adds `--layer_widths 4,4 --depth 4 --max_edge 48 --max_prob_tokens 600`.

## Tables and figure

With evaluation outputs laid out as `<root>/<model>/op_<N>.json` and
`<root>/<model>/in_dist_mixture/seed<k>.json`, the commands are as follows.

| paper item | command |
|---|---|
| answer accuracy | `python analysis/summarize.py <root>` |
| validity x correctness | `python analysis/quadrants.py <root>/<model>` |
| validity against op (figure) | `python analysis/plot_pvc.py <root>/<model>` |
| causes of invalid traces behind correct answers | `python analysis/ic_causes.py <root>/<model>` |
| minimality | `python analysis/minimality.py <root>/<model>` |
| re-ask | `python analysis/reask_analysis.py --evals_dir <dir>` |
| example traces per cell | `python analysis/cell_examples.py <root>/<model>/op_23.json --op 23` |

## Trained models

Every model in the paper is at https://huggingface.co/ratishsp/igsm-trace-validity,
one folder per model.

```python
from transformers import GPTNeoXForCausalLM
model = GPTNeoXForCausalLM.from_pretrained("ratishsp/igsm-trace-validity", subfolder="clean-run-a")
```

## Citation

```bibtex
@article{puduppully2026correct,
  title   = {Correct Answers, Invalid Traces: What Verifiable Grade-School Math Reveals About Chain-of-Thought Traces},
  author  = {Puduppully, Ratish and Misra, Pranabendu and Iyer, Paarth and Kalwar, Durgesh and Palod, Vardhan and Kambhampati, Subbarao},
  journal = {arXiv preprint arXiv:2609.38107},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.38107}
}
```

## License

MIT. The iGSM generator is a separate repository under its own license.
