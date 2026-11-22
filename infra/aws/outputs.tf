output "status" {
  value = var.enabled ? "ON: billing about $0.80/hour (on demand) until you turn it off or it auto-terminates after ${var.max_hours} h" : "OFF: no GPU resources exist"
}

output "gateway_url" {
  description = "Set this as VIZWIZ_GATEWAY_URL in Vercel (or the dashboard's environment)."
  value       = var.enabled ? module.gpu_server[0].gateway_url : null
}

output "instance_id" {
  value = var.enabled ? module.gpu_server[0].instance_id : null
}

output "setup_log_command" {
  description = "Follow the server's setup (needs the AWS CLI Session Manager plugin)."
  value       = var.enabled ? "aws ssm start-session --region ${var.region} --target ${module.gpu_server[0].instance_id}  # then: sudo tail -f /var/log/vizwiz-setup.log" : null
}
