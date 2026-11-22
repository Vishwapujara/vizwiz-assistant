# AWS GPU server (off by default)

Terraform for one **g6.xlarge** (1 × NVIDIA L4, the GPU every number in this project
was measured on) running vLLM and the gateway, so the dashboard's **Run live** button
can reach the real fine-tuned model. Decision: D-069.

**Off is the default and costs $0.** Everything billable sits behind `enabled`; with
`enabled = false` it does not exist. There is no load balancer, NAT gateway or Elastic
IP, because those bill even when idle. The only things kept while off are two free SSM
parameters and, if you ask for it, a free budget alert.

| | Cost |
|---|---|
| Off | $0 |
| On, on demand | about $0.80/hour (us-east-1) + a few cents of disk and public IP |
| On, `use_spot = true` | about $0.30–0.40/hour; AWS may reclaim it at any time |

Safety net: the server terminates itself after `max_hours` (default 2) even if you
forget to turn it off. Run `terraform apply -var enabled=false` afterwards to tidy up.

## One-time setup

1. **Install** the AWS CLI and sign in: `aws configure` (or `aws configure sso`).
2. **Ask for GPU quota.** New accounts usually allow 0 GPU instances. In the AWS
   console → Service Quotas → Amazon EC2, request **4 vCPUs** for *Running On-Demand
   G and VT instances* (or *All G and VT Spot Instance Requests* for spot). Approval
   takes from minutes to a couple of days.
3. **Store the two secrets** (free standard SSM parameters, encrypted):

   ```bash
   aws ssm put-parameter --name /vizwiz/gateway-api-key --type SecureString \
     --value "$(openssl rand -hex 24)"
   aws ssm put-parameter --name /vizwiz/wandb-api-key --type SecureString \
     --value "<your W&B API key>"      # the E5 adapter comes from the W&B registry
   ```

4. `cd infra/aws && terraform init`, then optionally
   `cp terraform.tfvars.example terraform.tfvars` and edit it.

## Turn it on

```bash
terraform apply -var enabled=true        # ~1 min to create, then ~10–15 min of setup
terraform output gateway_url              # e.g. http://3.91.x.x:8080
curl "$(terraform output -raw gateway_url)/health"   # {"status":"ok",...} when ready
```

At boot the server installs the pinned packages from this repo, downloads
Qwen3-VL-4B from Hugging Face, and the gateway pulls the `production` E5 adapter from
the W&B registry. Follow the progress with the `setup_log_command` output.

To let the dashboard call it, set two environment variables (in Vercel: Project →
Settings → Environment Variables, then redeploy; locally: in `dashboard/.env.local`):

```
VIZWIZ_GATEWAY_URL=http://<public ip>:8080
VIZWIZ_API_KEY=<the value stored in /vizwiz/gateway-api-key>
```

The public IP changes every time the server is created, so update
`VIZWIZ_GATEWAY_URL` after each turn-on.

## Turn it off

```bash
terraform apply -var enabled=false       # or: terraform destroy
```

Check the console shows no running `vizwiz-gpu` instance.

## Security

- Port 8080 is the only way in; `POST /v1/assist` needs the `X-API-Key` header.
  Narrow `allowed_cidrs` to block everyone else (that also blocks Vercel's live calls).
- No SSH and no key pairs: shell access is through Session Manager.
- Secrets live in SSM, are read at boot into a root-only file, and never reach the
  Terraform state or the setup log.
- Traffic is plain HTTP to the instance. The dashboard calls it from its server, not
  from the visitor's browser, and only with the five sample photos.
