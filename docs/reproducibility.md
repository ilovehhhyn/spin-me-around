# Reproducibility guide

## 1. Create an isolated environment

```bash
python -m venv .venv
source .venv/bin/activate
make install
make check
```

The supported Python versions are 3.11 and 3.12. CI checks both versions and verifies that the
package can be built from `pyproject.toml`.

## 2. Run the experiment in stages

Start with the smoke configuration:

```bash
make smoke
```

Then run the single-seed common-support pilot for full fine-tuning versus rank-8 LoRA:

```bash
make rank-pilot
```

Only after inspecting that pilot should you launch the confirmatory matrix:

```bash
dnd-sweep --config configs/week1_full.yaml
```

The first run downloads MNIST and FashionMNIST under the ignored `data/` directory. Generated
checkpoints and measurements are written beneath the configured ignored `runs/` directory.

## 3. Preserve the evidence needed to reproduce a run

Every run records its fully resolved configuration, Git commit and dirty status, UTC timestamp,
package version, Python version, platform, PyTorch version, and NumPy version. Treat a dirty-tree
run as exploratory: commit the code and rerun before using it as paper evidence.

The harness seeds Python, NumPy, and PyTorch and disables cuDNN benchmarking. GPU kernels can still
have platform-specific numerical variation, so confirm headline comparisons over the configured
three seeds rather than expecting bit-for-bit equality across hardware.

## 4. Compare full fine-tuning and LoRA fairly

Do not compare methods at an arbitrarily shared learning rate. First identify common support in
new-task performance and KL, then compare forgetting, bits, probes, and recovery within that
matched region. Record exclusions and matching tolerances before inspecting final outcomes.

The rank pilot is an integration and support-finding stage, not confirmatory evidence. The detailed
scientific protocol is in [`research/proposal.md`](research/proposal.md), and the Week-1 operational
handoff is in [`runbooks/week1-implementation.md`](runbooks/week1-implementation.md).

## 5. Provisioning Azure compute

Infrastructure definitions live only in `infra/azure/`. Follow its README to verify family quota,
initialize Terraform, and create a saved plan for review. Do not run `terraform apply` as part of
the reproducibility workflow; provisioning is a separate, explicitly authorized operation.
