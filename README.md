# Distance Is Not Damage

The revised proposal is in
[`docs/research/proposal.md`](docs/research/proposal.md).


## Repository guide

- [`docs/architecture.md`](docs/architecture.md) explains module boundaries and the run-artifact
  contract.
- [`docs/reproducibility.md`](docs/reproducibility.md) gives the staged smoke, pilot, and
  confirmatory workflow.
- [`configs/`](configs) contains version-controlled experiment definitions.
- [`src/distance_not_damage/`](src/distance_not_damage) is the installable research package;
  [`tests/`](tests) contains its fast contract tests.
- [`infra/azure/`](infra/azure) contains the isolated, plan-first GPU VM infrastructure.

## we are currently at: week 2.5

## Azure

Terraform for one Azure GPU VM lives in [`infra/azure`](infra/azure). It references an existing
resource group and cannot create resources outside that group. Follow the quota gate in its README;
never run `apply` until the saved plan has been reviewed.
