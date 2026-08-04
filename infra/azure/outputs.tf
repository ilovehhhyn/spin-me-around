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
