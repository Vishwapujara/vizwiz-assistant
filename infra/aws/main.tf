# VizWiz GPU server on AWS (D-069). Off by default: with enabled = false, nothing
# billable exists. See README.md for the on/off commands.

module "gpu_server" {
  source = "./modules/gpu-server"
  count  = var.enabled ? 1 : 0

  region              = var.region
  instance_type       = var.instance_type
  use_spot            = var.use_spot
  max_hours           = var.max_hours
  allowed_cidrs       = var.allowed_cidrs
  repo_url            = var.repo_url
  repo_ref            = var.repo_ref
  api_key_parameter   = var.api_key_parameter
  wandb_key_parameter = var.wandb_key_parameter
  root_volume_gb      = var.root_volume_gb
  ami_name_pattern    = var.ami_name_pattern
}

# Optional cost alarm for the whole account. AWS Budgets are free (first two per account).
resource "aws_budgets_budget" "monthly" {
  count        = var.budget_email == "" ? 0 : 1
  name         = "vizwiz-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_email]
  }
}
