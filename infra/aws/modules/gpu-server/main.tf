# One g6.xlarge (NVIDIA L4) running vLLM + the gateway, set up from the repo at boot.
# No load balancer, NAT gateway or Elastic IP: they bill even when idle.

data "aws_caller_identity" "me" {}
data "aws_partition" "current" {}

data "aws_vpc" "default" {
  default = true
}

# Pick a default-VPC subnet in an availability zone that actually offers the GPU type.
data "aws_ec2_instance_type_offerings" "gpu" {
  location_type = "availability-zone"
  filter {
    name   = "instance-type"
    values = [var.instance_type]
  }
}

data "aws_subnets" "usable" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "availability-zone"
    values = data.aws_ec2_instance_type_offerings.gpu.locations
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = [var.ami_name_pattern]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

resource "aws_security_group" "gateway" {
  name_prefix = "vizwiz-gateway-"
  description = "VizWiz gateway: port 8080 in (X-API-Key required), everything out"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "Gateway API"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }

  egress {
    description = "Package, model and adapter downloads"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Instance role: Session Manager access (no SSH keys, no port 22) and the two secrets.
resource "aws_iam_role" "server" {
  name_prefix = "vizwiz-gpu-"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.server.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "read_secrets" {
  name = "read-vizwiz-secrets"
  role = aws_iam_role.server.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "ssm:GetParameter"
        Resource = [for p in [var.api_key_parameter, var.wandb_key_parameter] :
        "arn:${data.aws_partition.current.partition}:ssm:${var.region}:${data.aws_caller_identity.me.account_id}:parameter${p}"]
      },
      {
        Effect    = "Allow"
        Action    = "kms:Decrypt"
        Resource  = "*"
        Condition = { StringEquals = { "kms:ViaService" = "ssm.${var.region}.amazonaws.com" } }
      },
    ]
  })
}

resource "aws_iam_instance_profile" "server" {
  name_prefix = "vizwiz-gpu-"
  role        = aws_iam_role.server.name
}

resource "aws_instance" "gpu" {
  ami                    = data.aws_ami.dlami.id
  instance_type          = var.instance_type
  subnet_id              = data.aws_subnets.usable.ids[0]
  vpc_security_group_ids = [aws_security_group.gateway.id]
  iam_instance_profile   = aws_iam_instance_profile.server.name

  associate_public_ip_address = true
  # The auto-off timer powers the machine down; "terminate" means it stops billing
  # for compute and disk alike, instead of lingering as a stopped instance.
  instance_initiated_shutdown_behavior = "terminate"

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        spot_instance_type             = "one-time"
        instance_interruption_behavior = "terminate"
      }
    }
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_gb
    encrypted             = true
    delete_on_termination = true
  }

  metadata_options {
    http_tokens = "required" # IMDSv2 only
  }

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    region              = var.region
    repo_url            = var.repo_url
    repo_ref            = var.repo_ref
    api_key_parameter   = var.api_key_parameter
    wandb_key_parameter = var.wandb_key_parameter
    max_minutes         = var.max_hours * 60
  })
  user_data_replace_on_change = true

  tags = { Name = "vizwiz-gpu" }

  lifecycle {
    precondition {
      condition     = length(data.aws_subnets.usable.ids) > 0
      error_message = "No default-VPC subnet in an availability zone that offers ${var.instance_type} in ${var.region}. Try another region."
    }
  }
}
