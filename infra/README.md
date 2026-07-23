# infra/ — Terraform for the AWS benchmark pipeline

Workload-layer IaC for running ratel-bench on CodeBuild in the **benchmarks
account (`340303437758`, `eu-central-1`)**. Follows the org-layer conventions
from [ratel-infra](https://github.com/ratel-ai/ratel-infra) (shared tfstate
bucket, assume-role into the member account, aws provider `~> 5.0`). Operating
the pipeline: [docs/aws-runbook.md](../docs/aws-runbook.md).

## What it creates (all zero-idle-cost)

- **ECR** `ratel-bench` — slim arm64 toolchain image (built/pushed manually
  from the repo `Dockerfile`; no repo code baked)
- **S3** `ratel-bench-results-340303437758` — versioned, SSE; mirrors
  `results/` (raw cells + canonical caches + reports) and `test-data/`
- **CodeBuild** `ratel-bench` — ARM/Graviton, GitHub source via
  CodeConnections, per-run env vars pre-filled with defaults, 8 h timeout,
  local cargo/pnpm cache
- **IAM** `ratel-bench-codebuild` — logs, ECR pull, S3 rw (bucket-scoped),
  `ssm:GetParameter` on `/ratel/bench/*`, `bedrock:InvokeModel*` on inference
  profiles + Anthropic foundation models
- **SSM SecureString** placeholders `/ratel/bench/{OPENAI_API_KEY,
  RATEL_DEPLOY_KEY, ANTHROPIC_API_KEY, MODEL_API_KEY}` — real values are set
  out-of-band with `aws ssm put-parameter --overwrite` (never in tfstate);
  Terraform ignores value drift
- **CodeStar/CodeConnections** GitHub connection + CloudWatch log group

## Applying

```bash
aws sso login --profile ratel-admin
terraform -chdir=infra init
terraform -chdir=infra apply
```

State: `s3://ratel-tfstate-288742312794/ratel-bench/terraform.tfstate`
(management account; `use_lockfile`). The provider assumes
`OrganizationAccountAccessRole` into the benchmarks account, so `ratel-admin`
credentials suffice.

## Two manual one-time steps Terraform cannot do

1. **GitHub connection handshake** — the connection is created `PENDING`;
   authorize it in the console (Developer Tools → Connections) against the
   **ratel-ai org's** "AWS Connector for GitHub" app installation, then register
   it as the account's CodeBuild credential (an inline project `auth{}` block is
   rejected by the CreateProject API):

   ```bash
   aws codebuild import-source-credentials --profile ratel-bench --region eu-central-1 \
     --server-type GITHUB --auth-type CODECONNECTIONS --token <connection-arn>
   ```

2. **Bedrock Anthropic use-case form** — once per account, else Claude invokes
   404 (see runbook Troubleshooting).
