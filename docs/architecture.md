# Repository architecture

The repository follows a conventional Python research-project layout: importable code lives under
`src/`, tests mirror observable behavior, experiment definitions are version-controlled YAML, and
generated artifacts are excluded from Git.

```text
spin-me-around/
├── .github/workflows/       # Pull-request and main-branch CI
├── configs/                 # Named, reviewable experiment matrices
├── docs/
│   ├── research/            # Proposal and scientific design review
│   └── runbooks/            # Operational experiment instructions
├── infra/azure/             # Isolated, plan-first Azure Terraform
├── src/distance_not_damage/ # Installable training and measurement package
├── tests/                   # Fast unit and contract tests
├── CONTRIBUTING.md
├── Makefile                 # Stable developer and experiment commands
└── pyproject.toml           # Build, dependency, and tool configuration
```

## Runtime boundaries

| Module | Responsibility |
| --- | --- |
| `config.py` | Validate YAML inputs and produce the fully resolved run configuration. |
| `data.py` | Construct deterministic datasets and loaders. |
| `model.py` | Define the task-conditioned MLP and full/LoRA parameterizations. |
| `objectives.py` | Implement supervised and reinforcement-learning objectives. |
| `training.py` | Execute pretraining and fine-tuning loops and checkpoint callbacks. |
| `metrics.py` | Measure accuracy, KL, code length, parameter distance, CKA, and probes. |
| `experiment.py` | Orchestrate one run or a sequential sweep and atomically write artifacts. |
| `provenance.py` | Capture source revision and runtime versions for auditability. |
| `cli.py` | Expose `dnd-train` and `dnd-sweep` command-line entry points. |

The dependency direction is intentionally one-way: the CLI loads configuration, the experiment
runner composes data/model/training/metrics, and lower-level modules do not know about output
directories or sweep orchestration.

## Run artifact contract

Each completed run directory contains:

- `resolved_config.json`: every effective setting after YAML defaults are applied;
- `provenance.json`: schema version, UTC creation time, package/runtime versions, and Git state;
- `metrics.jsonl`: ordered step-level measurements;
- `final_model.pt`: final model state;
- `summary.json`: completion marker and final result summary.

Writes use a temporary file followed by an atomic replacement. An existing incomplete run is never
silently overwritten; it must be inspected and moved aside first.

Base checkpoints are cached separately by a hash of the pretraining-relevant configuration. Sweep
runs are deliberately sequential, giving datasets, caches, and result paths a single writer.
