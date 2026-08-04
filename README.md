# Distance Is Not Damage

Week-1 research harness for reproducing the ParityMNIST/FashionMNIST result from
[RL's Razor](https://arxiv.org/abs/2509.04259) while crossing training objective with update
parameterization: full fine-tuning versus all-linear-layer LoRA rank 8.

The revised proposal is in
[`outputs/distance-is-not-damage-proposal.md`](outputs/distance-is-not-damage-proposal.md).
The corresponding methodological critique is in
[`outputs/design-review.md`](outputs/design-review.md), and implementation/Azure verification status
is recorded in [`outputs/week1-implementation-handoff.md`](outputs/week1-implementation-handoff.md).

## What the harness implements

- The published `785 -> 512 -> 256 -> 10` MLP with a binary task indicator.
- Joint pretraining on deterministic 500-example subsets of MNIST and FashionMNIST.
- Parity SFT-1, SFT-2, base-conditioned oracle SFT, 1–0 REINFORCE, GRPO, and GRPO with KL.
- Native LoRA for the custom MLP with frozen base weights, configurable rank/alpha/dropout/targets,
  zero-output initialization, and effective-weight comparisons against full fine-tuning.
- Forward and reverse KL, task accuracy/forgetting, effective update norms, linear CKA, old-task
  label code length in bits, and fixed/fresh linear probes.
- Deterministic JSONL checkpoint records and a sequential sweep runner.
- Float32 training throughout; no mixed-precision path is present in Week 1.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

Run the first configured smoke experiment, or select a cell explicitly:

```bash
dnd-train --config configs/week1_smoke.yaml
dnd-train --config configs/week1_smoke.yaml \
  --method sft_1 --parameterization lora --learning-rate 0.001
```

Run the smoke sweep:

```bash
dnd-sweep --config configs/week1_smoke.yaml
```

The first run downloads MNIST and FashionMNIST into the ignored `data/` directory. Results and
checkpoints are written beneath the configured ignored `runs/` directory.

## Week-1 rank pilot and full sweep

`configs/week1_rank_pilot.yaml` is the intended first scientific pilot. It contains the four-cell
SFT-1/REINFORCE × full/rank-8 design, five independently chosen learning rates per
parameterization, one scheduler, one epoch, and one seed. Its purpose is to find common support
before spending three-seed compute:

```bash
dnd-sweep --config configs/week1_rank_pilot.yaml
```

`configs/week1_full.yaml` contains the paper's 15 log-spaced learning rates, both schedulers, and
one/two-epoch settings for both full fine-tuning and LoRA. LoRA uses a separate 10×-shifted grid;
the correct comparison is made after matching task learning and KL, not at an arbitrarily shared
learning rate. Begin with the smoke and rank-pilot configs before launching this 2,160-run grid.
Each run is executed sequentially so result files and dataset downloads have a single writer.

The full confirmatory replication uses three seeds. The single-seed pilot and smoke values are
integration diagnostics, not research evidence.

## Azure

Terraform for one Azure GPU VM lives in [`infra/azure`](infra/azure). It references an existing
resource group and cannot create resources outside that group. Follow the quota gate in its README;
never run `apply` until the saved plan has been reviewed.
