# Azure GPU VM: plan-only workflow

This directory defines one GPU VM and its dedicated network resources. It reads the existing
`august-2026-1` resource group and creates every resource inside it. Variable validation pins both
the resource group and region so an edited `.tfvars` file cannot silently broaden scope.

The provider constraint is `hashicorp/azurerm ~> 5.0`. The configuration uses the v5.0.1 resource
schemas for `azurerm_linux_virtual_machine`, networking, and the existing resource-group data
source. The VM image is `microsoft-dsvm:ubuntu-hpc:2204:latest`; the OS disk is 512 GB
`Premium_LRS`.

## Current execution status

Configuration generation and offline provider-schema validation can be performed without Azure
credentials. The quota command and `terraform plan` require an authenticated Azure CLI session.
Do not run `terraform plan` until the quota gate below passes. Never run `terraform apply` from an
unreviewed plan, and never run `terraform destroy` without explicit confirmation.

## 1. Supply local values

```bash
cp terraform.tfvars.example terraform.tfvars
```

Edit the ignored copy. The requested public key must already exist; Terraform never generates or
stores a private key. Set `ssh_source_cidr` to the operator's current public address with `/32`.
Open SSH access from `0.0.0.0/0` is rejected by variable validation.

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
one VNet, one subnet, one NSG, and one subnet/NSG association—all under `august-2026-1`. The
resource group itself must appear only as a data source.
