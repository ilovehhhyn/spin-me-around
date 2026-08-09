variable "subscription_id" {
  description = "Azure subscription containing the existing resource group."
  type        = string
  sensitive   = true
  nullable    = false
}

variable "tenant_id" {
  description = "Microsoft Entra tenant used by the AzureRM provider."
  type        = string
  sensitive   = true
  nullable    = false
}

variable "resource_group_name" {
  description = "Existing resource group. This project is pinned to the authorized group."
  type        = string
  default     = "august-2026-1"

  validation {
    condition     = var.resource_group_name == "august-2026-1"
    error_message = "This configuration is authorized only for resource group august-2026-1."
  }
}

variable "region" {
  description = "Azure region for every managed resource."
  type        = string
  default     = "westus2"

  validation {
    condition     = var.region == "westus2"
    error_message = "This configuration is pinned to westus2 for the reviewed deployment."
  }
}

variable "vm_size" {
  description = "GPU VM SKU. Check its regional family quota before planning."
  type        = string
  default     = "Standard_NC40ads_H100_v5"
}

variable "vm_name" {
  description = "Name and hostname of the training VM."
  type        = string
  default     = "dnd-week2-gpu"

  validation {
    condition     = can(regex("^[a-zA-Z0-9][a-zA-Z0-9-]{0,62}$", var.vm_name))
    error_message = "vm_name must be 1-63 alphanumeric or hyphen characters."
  }
}

variable "experiment_phase" {
  description = "Experiment phase recorded on every managed resource tag."
  type        = string
  default     = "week2-phase0"

  validation {
    condition     = length(trimspace(var.experiment_phase)) > 0
    error_message = "experiment_phase cannot be empty."
  }
}

variable "admin_username" {
  description = "Linux administrator account used for SSH."
  type        = string
  default     = "azureuser"
}

variable "ssh_public_key_path" {
  description = "Local path to an existing OpenSSH public key."
  type        = string
  nullable    = false

  validation {
    condition     = endswith(var.ssh_public_key_path, ".pub")
    error_message = "ssh_public_key_path must point to a .pub file."
  }
}

variable "ssh_source_cidr" {
  description = "Single operator public IPv4 CIDR allowed to reach TCP/22. Prefer x.x.x.x/32."
  type        = string
  nullable    = false

  validation {
    condition     = can(cidrhost(var.ssh_source_cidr, 0)) && var.ssh_source_cidr != "0.0.0.0/0"
    error_message = "ssh_source_cidr must be a valid, restricted CIDR; 0.0.0.0/0 is forbidden."
  }
}

variable "auto_shutdown_enabled" {
  description = "Whether Azure should automatically stop the VM each day."
  type        = bool
  default     = true
}

variable "auto_shutdown_time" {
  description = "Daily automatic shutdown time in Azure HHmm format."
  type        = string
  default     = "2300"

  validation {
    condition     = can(regex("^(?:[01][0-9]|2[0-3])[0-5][0-9]$", var.auto_shutdown_time))
    error_message = "auto_shutdown_time must be a valid 24-hour HHmm value."
  }
}

variable "auto_shutdown_timezone" {
  description = "Azure time-zone identifier used by the shutdown schedule."
  type        = string
  default     = "Pacific Standard Time"

  validation {
    condition     = length(trimspace(var.auto_shutdown_timezone)) > 0
    error_message = "auto_shutdown_timezone cannot be empty."
  }
}

variable "virtual_network_cidr" {
  description = "Address space for the dedicated training virtual network."
  type        = string
  default     = "10.42.0.0/16"
}

variable "subnet_cidr" {
  description = "Address prefix for the training subnet."
  type        = string
  default     = "10.42.1.0/24"
}
