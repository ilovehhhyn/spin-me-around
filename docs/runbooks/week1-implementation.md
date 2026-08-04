# Week-1 implementation handoff

**Updated August 3, 2026:** the harness now tests full fine-tuning versus all-layer LoRA rank 8
from Week 1, with bits and fixed/fresh probe diagnostics.

## Delivered

The workspace is now an initialized Git repository on branch `main` (no commit was created).

Research artifacts:

- `outputs/distance-is-not-damage-proposal.md` — revised proposal.
- `outputs/design-review.md` — comprehensive design critique and hypothesis-to-test audit.

Training code:

- `src/distance_not_damage/` — typed, class-based PyTorch package.
- `configs/week1_smoke.yaml` — four-cell SFT-1/REINFORCE × full/rank-8 integration run.
- `configs/week1_rank_pilot.yaml` — focused 20-run common-support pilot.
- `configs/week1_full.yaml` — separate 15-rate full/LoRA grids, two schedulers, one/two epochs, six
  methods, and three seeds.
- `tests/` — objective, KL, oracle-distribution, CKA, and configuration tests.
- `pyproject.toml` — installable package and the `dnd-train` / `dnd-sweep` entry points.

Infrastructure:

- `infra/azure/main.tf`
- `infra/azure/variables.tf`
- `infra/azure/outputs.tf`
- `infra/azure/terraform.tfvars.example`
- `infra/azure/check_gpu_quota.sh`
- `infra/azure/README.md`

## Rank-8 implementation decisions

- The pretrained MLP is still learned with full parameters. LoRA is inserted only for task-B
  fine-tuning.
- Each linear layer uses $W=W_0+(\alpha/r)BA$ with $r=8$, $\alpha=8$, dropout zero, Kaiming
  initialization for $A$, and zero initialization for $B$.
- Base weights and biases are frozen. Effective deployed weights—not mismatched state-dict names—
  are used for update-distance metrics.
- Full and LoRA learning rates are swept independently. LoRA's configured grid is shifted 10×,
  then comparisons are made on common support in task-B performance and KL.
- A fixed FashionMNIST probe is fitted once on base representations. A fresh ridge probe is fitted
  at each checkpoint. Their gap is a reorientation diagnostic; neither probe is a literal bit count.
- Old-task code length is cross-entropy converted from nats to bits by division by $\ln 2$. It is a
  behavioral confidence diagnostic in this toy task, not a bits-per-weight capacity estimate.

The adapter is implemented directly rather than through a transformer PEFT library because this
Week-1 model is a three-layer custom MLP. The implementation preserves the same mathematical LoRA
parameterization without adding a transformer-specific dependency.

## Verification performed

- `ruff check src tests` — passed.
- `pytest` — 8 tests passed.
- Editable package installation — passed.
- `dnd-train --help` and `dnd-sweep --help` — passed.
- Four real-data full/rank-8 × SFT/REINFORCE smoke cells — passed end to end on MPS.
- Zero-step equivalence in all four cells — exact: max logit error, KL, effective parameter distance,
  and code-length change were all 0; fixed and fresh probes were identical.
- Terraform 1.15.8 `fmt -check` — passed after formatting.
- Terraform `init -backend=false` — selected signed `hashicorp/azurerm` v5.0.1.
- Terraform `validate` — passed.
- Shell syntax check for the quota gate — passed.
- Git ignore audit — real `.tfvars`, `.tfstate*`, `.tfplan`, and `.terraform/` are ignored.

The smoke run is an integration test, not a replication or rank result. With only three
base-pretraining epochs, 2,000 fine-tuning examples, one seed, and unmatched endpoints, it produced:

| Method | Update | Parity | Fashion | Forgetting | Forward KL | A bits lost/example | Fixed probe | Fresh probe |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SFT-1 | Full | 0.746 | 0.354 | 0.215 | 0.431 | 1.373 | 0.125 | 0.736 |
| SFT-1 | LoRA $r=8$ | 0.713 | 0.283 | 0.287 | 0.640 | 2.299 | 0.109 | 0.737 |
| 1–0 REINFORCE | Full | 0.767 | 0.498 | 0.072 | 0.047 | 0.133 | 0.344 | 0.731 |
| 1–0 REINFORCE | LoRA $r=8$ | 0.740 | 0.531 | 0.039 | 0.066 | 0.209 | 0.535 | 0.737 |

The full model has 536,330 trainable parameters; rank-8 LoRA has 18,648. Do not compare the table's
full and LoRA rows causally: the task-B accuracies and KL values are not matched. First run the
independent learning-rate grids in `week1_rank_pilot.yaml`, identify common support, then repeat the
supported region across seeds.

## Azure status: stopped before plan

No Azure resources were changed. `terraform plan`, `apply`, and `destroy` were not run.

The Azure CLI was installed, but device authentication timed out. Therefore the required
`az vm list-usage --location westus2 -o table` command has not run and the GPU quota is unknown.

The supplied public-key path, `/Users/Helen/.ssh/id_ed25519_spark.pub`, does not exist on this
machine. No alternative `.pub` key exists under the current user's `.ssh` directory. Terraform
would fail when reading the key, so planning remains blocked even after quota authentication.

The supplied subscription and tenant identifiers are stored only in ignored
`infra/azure/terraform.tfvars`. They do not appear in the tracked-file set. The current operator IP
was stored as a `/32` SSH source rule in that same ignored file; `0.0.0.0/0` is rejected by variable
validation.

## Actions needed to unblock planning

1. Put the intended OpenSSH public key at the supplied path, or update the ignored
   `ssh_public_key_path` to the correct existing `.pub` file.
2. Authenticate:

   ```bash
   az login --tenant bc4ac058-e074-4f4d-a7ab-f8f4d8af3472
   ```

3. In `infra/azure`, run the quota gate:

   ```bash
   ./check_gpu_quota.sh \
     257a4cdc-7a4c-44bd-a28d-5bc4c31721cf \
     westus2 \
     'NCads H100 v5' \
     40
   ```

4. If and only if the family quota is nonzero and at least 40 vCPUs remain, verify regional SKU and
   image availability, then run `terraform plan -out=training-vm.tfplan` and stop for review.

The selected default is `Standard_NC40ads_H100_v5`, a one-H100, 40-vCPU SKU. Quota does not
guarantee physical allocation capacity in West US 2. Azure's published Ubuntu-HPC image support
list does not currently include this NCads family, so the image/SKU preflight is also a hard gate;
the requested pairing must not be assumed compatible merely because both names are valid.
