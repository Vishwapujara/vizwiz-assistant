variable "enabled" {
  description = "Master switch. false (default) = no GPU server and nothing billable exists. true = create it."
  type        = bool
  default     = false
}

variable "region" {
  description = "AWS region. g6 (NVIDIA L4) instances must be offered there."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "One NVIDIA L4 (24 GB), the GPU every number in this project was measured on."
  type        = string
  default     = "g6.xlarge"
}

variable "use_spot" {
  description = "Spot capacity: roughly 60% cheaper, but AWS can reclaim the instance at any time."
  type        = bool
  default     = false
}

variable "max_hours" {
  description = "Safety net: the instance powers off and terminates itself after this many hours, even if you forget it."
  type        = number
  default     = 2
  validation {
    condition     = var.max_hours >= 1 && var.max_hours <= 24
    error_message = "max_hours must be between 1 and 24."
  }
}

variable "allowed_cidrs" {
  description = "Who may reach the gateway on port 8080. Requests also need the X-API-Key header."
  type        = list(string)
  default     = ["0.0.0.0/0"] # Vercel's outbound addresses are not fixed
}

variable "repo_url" {
  description = "Git repository the server installs the gateway from."
  type        = string
  default     = "https://github.com/Vishwapujara/vizwiz-assistant.git"
}

variable "repo_ref" {
  description = "Branch or tag to install."
  type        = string
  default     = "main"
}

variable "api_key_parameter" {
  description = "SSM SecureString holding the gateway API key (create it once by hand, see README)."
  type        = string
  default     = "/vizwiz/gateway-api-key"
}

variable "wandb_key_parameter" {
  description = "SSM SecureString holding the W&B API key; the adapter comes from the W&B registry (production alias)."
  type        = string
  default     = "/vizwiz/wandb-api-key"
}

variable "root_volume_gb" {
  description = "Disk for Python packages (~10 GB) and the base model (~9 GB). Deleted with the instance."
  type        = number
  default     = 80
}

variable "ami_name_pattern" {
  description = "AWS Deep Learning AMI with NVIDIA drivers preinstalled (newest match is used)."
  type        = string
  default     = "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *"
}

variable "budget_email" {
  description = "If set, an AWS Budget (free) emails this address at 80% of budget_usd. Kept even when enabled = false."
  type        = string
  default     = ""
}

variable "budget_usd" {
  description = "Monthly budget for the alert."
  type        = number
  default     = 20
}
