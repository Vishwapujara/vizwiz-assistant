output "gateway_url" {
  value = "http://${aws_instance.gpu.public_ip}:8080"
}

output "instance_id" {
  value = aws_instance.gpu.id
}
