terraform {
  required_version = ">= 1.10.0, < 2.0.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 5.0"
    }
  }
}

provider "azurerm" {
  features {}

  subscription_id = var.subscription_id
  tenant_id       = var.tenant_id
}

# The resource group is intentionally read, not created. Every managed resource
# below names this existing group explicitly.
data "azurerm_resource_group" "training" {
  name = var.resource_group_name
}

locals {
  resource_prefix = var.vm_name
  common_tags = {
    managed-by = "terraform"
    project    = "distance-is-not-damage"
    purpose    = "ml-training"
  }
}

resource "azurerm_virtual_network" "training" {
  name                = "${local.resource_prefix}-vnet"
  address_space       = [var.virtual_network_cidr]
  location            = var.region
  resource_group_name = data.azurerm_resource_group.training.name
  tags                = local.common_tags
}

resource "azurerm_subnet" "training" {
  name                            = "${local.resource_prefix}-subnet"
  resource_group_name             = data.azurerm_resource_group.training.name
  virtual_network_name            = azurerm_virtual_network.training.name
  address_prefixes                = [var.subnet_cidr]
  default_outbound_access_enabled = true
}

resource "azurerm_network_security_group" "training" {
  name                = "${local.resource_prefix}-nsg"
  location            = var.region
  resource_group_name = data.azurerm_resource_group.training.name

  security_rule {
    name                       = "allow-ssh-from-operator"
    description                = "Allow SSH only from the operator's declared public CIDR."
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = var.ssh_source_cidr
    destination_address_prefix = "*"
  }

  tags = local.common_tags
}

resource "azurerm_subnet_network_security_group_association" "training" {
  subnet_id                 = azurerm_subnet.training.id
  network_security_group_id = azurerm_network_security_group.training.id
}

resource "azurerm_public_ip" "training" {
  name                = "${local.resource_prefix}-public-ip"
  location            = var.region
  resource_group_name = data.azurerm_resource_group.training.name
  allocation_method   = "Static"
  sku                 = "Standard"
  tags                = local.common_tags

  lifecycle {
    create_before_destroy = true
  }
}

resource "azurerm_network_interface" "training" {
  name                           = "${local.resource_prefix}-nic"
  location                       = var.region
  resource_group_name            = data.azurerm_resource_group.training.name
  accelerated_networking_enabled = true

  ip_configuration {
    name                          = "primary"
    primary                       = true
    subnet_id                     = azurerm_subnet.training.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.training.id
  }

  tags = local.common_tags
}

resource "azurerm_linux_virtual_machine" "training" {
  name                            = var.vm_name
  computer_name                   = var.vm_name
  location                        = var.region
  resource_group_name             = data.azurerm_resource_group.training.name
  size                            = var.vm_size
  admin_username                  = var.admin_username
  disable_password_authentication = true
  network_interface_ids           = [azurerm_network_interface.training.id]

  admin_ssh_key {
    username   = var.admin_username
    public_key = file(pathexpand(var.ssh_public_key_path))
  }

  os_disk {
    name                 = "${local.resource_prefix}-os-disk"
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
    disk_size_gb         = 512
  }

  source_image_reference {
    publisher = "microsoft-dsvm"
    offer     = "ubuntu-hpc"
    sku       = "2204"
    version   = "latest"
  }

  boot_diagnostics {}

  tags = local.common_tags

  depends_on = [azurerm_subnet_network_security_group_association.training]
}
