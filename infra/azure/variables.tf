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
  default     = "dnd-week1-gpu"

  validation {
    condition     = can(regex("^[a-zA-Z0-9][a-zA-Z0-9-]{0,62}$", var.vm_name))
    error_message = "vm_name must be 1-63 alphanumeric or hyphen characters."
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
