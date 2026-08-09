---
covers:
- infra/azure/main.tf
- infra/azure/outputs.tf
- infra/azure/terraform.tfvars.example
- infra/azure/variables.tf
---
# Azure GPU VM: plan-only workflow

This directory defines one GPU VM and its dedicated network resources. It reads the existing
`august-2026-1` resource group and creates every resource inside it. Variable validation pins both
the resource group and region so an edited `.tfvars` file cannot silently broaden scope.

The provider constraint is `hashicorp/azurerm ~> 5.0`. The configuration uses the v5.0.1 resource
schemas for `azurerm_linux_virtual_machine`, networking, and the existing resource-group data
source. The VM image is `microsoft-dsvm:ubuntu-hpc:2204:latest`; the OS disk is 512 GB
`Premium_LRS`.

The configuration targets the Week-2 Phase-0 sweep. The default VM name is `dnd-week2-gpu`, and
every managed resource carries a `phase` tag set from `experiment_phase`, which defaults to
`week2-phase0`.

## Current execution status

Configuration generation and offline provider-schema validation can be performed without Azure
credentials. The quota command and `terraform plan` require an authenticated Azure CLI session.
Do not run `terraform plan` until the quota gate below passes. Never run `terraform apply` from an
unreviewed plan, and never run `terraform destroy` without explicit confirmation.

`terraform init` with AzureRM 5.0.1, `terraform fmt -check`, and `terraform validate` pass.

### Open blocker: Microsoft.Compute is NotRegistered

The authenticated subscription and the existing `westus2` resource group are reachable, but the
`Microsoft.Compute` resource provider is `NotRegistered`. The required `az vm list-usage` table
returned no quota records, so the quota gate did not pass and no plan was run.

Provider registration is subscription-scoped and falls outside the authorized
resource-group-only mutation boundary. You or a subscription administrator must register
`Microsoft.Compute`, then rerun the quota gate and the SKU and image gates below before planning.

## 1. Supply local values

```bash
cp terraform.tfvars.example terraform.tfvars
```

Edit the ignored copy. The requested public key must already exist; Terraform never generates or
stores a private key. Set `ssh_source_cidr` to the operator's current public address with `/32`.
Open SSH access from `0.0.0.0/0` is rejected by variable validation.

### Automatic shutdown guard

GPU instances are expensive and easy to leave idle, so the configuration creates a daily Azure VM
shutdown schedule that is enabled by default and configurable for deliberately overnight
experiments. Shutdown notifications are disabled.

| Variable | Type | Default | Notes |
| --- | --- | --- | --- |
| `auto_shutdown_enabled` | `bool` | `true` | Whether Azure automatically stops the VM each day. |
| `auto_shutdown_time` | `string` | `"2300"` | Daily shutdown time in Azure `HHmm` format. Validation rejects any value that is not a valid 24-hour `HHmm` value. |
| `auto_shutdown_timezone` | `string` | `"Pacific Standard Time"` | Azure time-zone identifier used by the schedule. Cannot be empty. |
| `experiment_phase` | `string` | `"week2-phase0"` | Experiment phase recorded on every managed resource tag. Cannot be empty. |

Changes take effect on the next apply of a reviewed plan. The `automatic_shutdown` output reports
the configured `enabled`, `time`, and `timezone` values.

## 2. Authenticate and check quota

```bash
az login --tenant <tenant-id>
az account set --subscription <subscription-id>
./check_gpu_quota.sh \
  <subscription-id> \
  westus2 \
  'NCads H100 v5' \
  40
```

The script first runs the required table-form `az vm list-usage` command, then selects the quota
record and stops if the family limit is zero or fewer than 40 vCPUs remain. If Azure names the
quota family differently in this subscription, rerun with the exact family fragment shown in the
table. Quota sufficiency does not guarantee physical H100 capacity in a region or zone.

Also confirm the SKU and image are offered in the target subscription/region:

```bash
az vm list-skus --location westus2 --size Standard_NC40ads_H100_v5 --all --output table
az vm image show --location westus2 --urn microsoft-dsvm:ubuntu-hpc:2204:latest
```

Azure's published Ubuntu-HPC support list does not currently name the NCads H100 v5 family, even
though it documents the image and SKU separately. Treat the image check as a hard compatibility
gate. If Azure rejects this pairing, choose an image explicitly offered for NCads H100 v5 or choose
an Ubuntu-HPC-supported GPU SKU; do not work around the check in Terraform.

## 3. Initialize, validate, and create a saved plan

Only after the quota gate passes:

```bash
terraform init
terraform fmt -check
terraform validate
terraform plan -out=training-vm.tfplan
terraform show training-vm.tfplan
```

Stop after reviewing the plan. The planned graph should contain one VM, one public IP, one NIC,
one VNet, one subnet, one NSG, one subnet/NSG association, and the daily shutdown schedule, all
under `august-2026-1`. The resource group itself must appear only as a data source.
