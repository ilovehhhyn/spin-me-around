# Contributing

This repository is an experimental research harness. Changes should preserve both software
correctness and the meaning of reported measurements.

## Development setup

Use Python 3.11 or 3.12 in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
make install
pre-commit install
```

Before opening or updating a pull request, run:

```bash
make check
make build
```

`make format` applies the repository's Ruff formatting and safe lint fixes.

## Experiment changes

When changing an experimental condition or measurement:

1. Add or update a version-controlled YAML configuration.
2. Add a focused unit test for the new behavior or invariant.
3. Update the proposal or runbook if the scientific interpretation changes.
4. Keep raw data, model weights, and run artifacts outside Git. Each run already records its
   resolved configuration and source/runtime provenance.

Do not silently change defaults that affect comparisons. Explain such changes in the pull request
and report old and new settings.

## Infrastructure safety

Azure Terraform is isolated under `infra/azure/`. It may reference only the designated existing
resource group. Run the quota check and `terraform plan`, save the plan locally, and stop for human
review. Never apply or destroy infrastructure without explicit authorization.

Never commit credentials, private SSH keys, `.tfvars`, Terraform state, model checkpoints, or
downloaded datasets. The pre-commit hooks include a private-key detector, but contributors remain
responsible for reviewing every staged change.
