output "public_ip" {
  description = "Public IPv4 address assigned to the training VM."
  value       = azurerm_public_ip.training.ip_address
}

output "ssh_command" {
  description = "SSH command using the private key paired with ssh_public_key_path."
  value = format(
    "ssh -i %s %s@%s",
    trimsuffix(pathexpand(var.ssh_public_key_path), ".pub"),
    var.admin_username,
    azurerm_public_ip.training.ip_address,
  )
}

output "automatic_shutdown" {
  description = "Configured daily VM shutdown guard."
  value = {
    enabled  = var.auto_shutdown_enabled
    time     = var.auto_shutdown_time
    timezone = var.auto_shutdown_timezone
  }
}
