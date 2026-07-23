# 10. AWS CodeBuild pipeline: cloud execution without behavior change

Date: 2026-07-23

## Status

Accepted. Moves benchmark execution from a developer laptop to a single AWS
CodeBuild job at minimum cost, keeping the fixed experiment design of
ADR-0005/0009 and `EXPERIMENTS.md` byte-identical for deterministic stages.
Operating guide: [../aws-runbook.md](../aws-runbook.md).

## Context

Every published number was produced on a laptop: hours-long occupancy, one
machine, one operator, no shared trigger. The dominant cost is LLM tokens
(already minimized by the version-independent control-arm cache — only
`ratel-full` runs live per version); compute is CPU-only and short-lived. The
org IaC ([ratel-infra](https://github.com/ratel-ai/ratel-infra)) provides a
dedicated benchmarks account, `eu-central-1` as default region, and a shared
tfstate bucket, and delegates workload infra to the workload repo.

## Decision

1. **Orchestration: one CodeBuild project** (`ARM_CONTAINER`, slim toolchain
   image from ECR, GitHub source via CodeConnections). No Step Functions, no
   ECS, no VPC, no idle infrastructure. `buildspec.yml` keeps the exact
   `EXPERIMENTS.md` commands visible, bookended by `version-set`/`version-reset`.
   One build == one (version, retriever); every per-run variable has a default
   (latest release), so a plain "Start build" is a valid benchmark run. The
   design constants (pools, top-k, scenarios, seed, arms) are hardcoded in the
   buildspec — changing them is a PR, not a console field.
2. **State: a versioned S3 bucket** mirrors `results/` (raw cells, canonical
   `agent.jsonl` control caches, append-only summaries, reports) plus
   `test-data/`. Pulled at `install`, pushed at `post_build`. Publishing stays
   a git operation: results are synced down, reviewed, and committed to `main`
   as before (the website rebuilds from `main`).
3. **LLM transport: Claude via Amazon Bedrock** (EU inference profiles,
   IAM-role auth — no Anthropic key in the cloud); `gpt-*` stays on the OpenAI
   API. Model ids on rows remain the friendly names, keeping pricing and
   report keys backend-independent. Because task-completion numbers shift
   slightly across serving stacks, control caches must be re-baselined once on
   Bedrock (`REBASELINE_CONTROLS=true` ⇒ `--force`) before Bedrock-era numbers
   are published; Bedrock-era and Anthropic-API-era Claude cells are not
   mix-and-match comparable.
4. **Models are endpoint links.** `RATEL_MODELS` accepts
   `{"models": ["<baseURL>#<model>", …]}`; adding a model is appending a link,
   with no code change. Well-known provider hosts (bedrock-runtime, Anthropic,
   OpenAI) resolve to native API routing; unknown hosts use the generic
   OpenAI-compatible client (the pre-existing user-hosted-endpoint path, e.g.
   qwen3-4b behind ratel-inference-gateway).
5. **Secrets: SSM Parameter Store SecureString (standard tier, $0)** — not
   Secrets Manager. The app keeps reading `agent/.env`; CI materializes it
   from SSM at `install`. Bedrock needs no secret at all.
6. **Trigger: manual** (console button or `aws codebuild start-build`), never
   scheduled — runs cost real money and follow releases, not calendars.
   `--dollar-global` (default $60/eval) caps each run's cloud spend.

## Consequences

- Idle cost ≈ $0.10/month (ECR image + S3 storage); a version run ≈ $0.50
  compute + ~$16 LLM tokens with cached controls.
- Anyone with account access can trigger and watch runs; the laptop is freed.
- Deterministic stages (retrieval eval, summarize, report) are byte-identical
  to laptop output — gate: diff container-produced rows against laptop rows
  before the first published cloud run.
- Two console one-timers exist outside Terraform (GitHub connection handshake +
  account-level source credential; Bedrock Anthropic use-case form) —
  documented in `infra/README.md` and the runbook.
- The toolchain image bakes no repo code or data, so code changes need no
  image rebuild; the GitHub clone is the source of truth per build.
