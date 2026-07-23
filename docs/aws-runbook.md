# AWS Runbook — running ratel-bench on CodeBuild

One build == one Ratel version benchmarked end-to-end at the fixed
[EXPERIMENTS.md](../EXPERIMENTS.md) design: retrieval eval (BFCL + SR-Agents) →
LLM eval (BFCL + SR-Agents) → summarize → report → both `report.json` to S3.
Deterministic stages are byte-identical to a laptop run; only the LLM transport
differs (Claude via Bedrock, IAM-role auth).

## Architecture (all zero-idle-cost)

```
Start build (console or CLI)                 benchmarks account 340303437758, eu-central-1
  └─ CodeBuild "ratel-bench" (ARM/Graviton, image from ECR, buildspec.yml)
       install:    .env from SSM · s3 sync caches+test-data · pnpm install
       pre_build:  version-reset → version-set <version under test> → SDK pin
       build:      retrieval eval → LLM eval   (Claude→Bedrock, gpt→OpenAI)
       post_build: summarize → report → version-reset → aws s3 sync results/
  └─ S3 ratel-bench-results-340303437758   (versioned; raw cells, caches, reports)
```

| Resource | Name |
|---|---|
| CodeBuild project | `ratel-bench` |
| Results bucket | `s3://ratel-bench-results-340303437758` |
| ECR image | `340303437758.dkr.ecr.eu-central-1.amazonaws.com/ratel-bench:latest` |
| Secrets (SSM SecureString, free tier) | `/ratel/bench/{OPENAI_API_KEY, RATEL_DEPLOY_KEY, ANTHROPIC_API_KEY, MODEL_API_KEY}` |
| Logs | CloudWatch `/aws/codebuild/ratel-bench` |
| IaC | [`infra/`](../infra) (Terraform; see its README) |

Idle cost ≈ $0.10/month (image + data storage). A run costs ~$0.50 compute +
LLM tokens (≈$16/version with cached controls; the `DOLLAR_GLOBAL` cap bounds it).

## Access

Console: sign in via SSO (https://d-81677d231b.awsapps.com/start) → management
account → **Switch role** → account `340303437758`, role
`OrganizationAccountAccessRole` → region **Frankfurt (eu-central-1)**.

CLI (`~/.aws/config`):

```ini
[sso-session ratel]
sso_start_url = https://d-81677d231b.awsapps.com/start
sso_region = eu-south-1
sso_registration_scopes = sso:account:access

[profile ratel-admin]
sso_session = ratel
sso_account_id = 288742312794
sso_role_name = PowerUserAccess
region = eu-central-1

[profile ratel-bench]
role_arn = arn:aws:iam::340303437758:role/OrganizationAccountAccessRole
source_profile = ratel-admin
region = eu-central-1
```

`aws sso login --profile ratel-admin` when the session expires; then use
`--profile ratel-bench` for everything in the benchmarks account.

## Trigger a run

Every variable has a default — a **plain "Start build" benchmarks the latest
release** (currently 0.4.0, bm25). Override only what differs.

**Console:** https://eu-central-1.console.aws.amazon.com/codesuite/codebuild/projects/ratel-bench
→ *Start build* (defaults) or *Start build with overrides* → Additional
configuration → edit the pre-filled environment variables → Start. Live logs
stream on the build page.

**CLI:**

```bash
# defaults (latest release, bm25):
aws codebuild start-build --profile ratel-bench --region eu-central-1 \
  --project-name ratel-bench

# different retriever:
aws codebuild start-build --profile ratel-bench --region eu-central-1 \
  --project-name ratel-bench \
  --environment-variables-override name=RETRIEVER,value=semantic

# different version (pre-0.4.0 generation):
aws codebuild start-build --profile ratel-bench --region eu-central-1 \
  --project-name ratel-bench \
  --environment-variables-override \
    name=RATEL_VERSION_ARGS,value="--tag v0.3.0-rc.1" \
    name=RATEL_EXPECT,value="0.3.0-rc.1" \
    name=RATEL_SDK_VERSION,value="0.2.2" \
    name=RATEL_GENERATION,value="pre-0.4.0"
```

### Per-run variables (all defaulted; visible pre-filled in the console)

| Variable | Default | Meaning |
|---|---|---|
| `RATEL_VERSION_ARGS` | `--crate 0.4.0` | `version-set` selector: `--crate <v>` \| `--tag <t>` \| `--rev <sha>` |
| `RATEL_EXPECT` | `0.4.0` | resolved version string `version-set` must assert |
| `RATEL_VERSION_LABEL` | `auto` | row label; `auto` derives `<version>-<sparse\|dense\|hybrid>` (0.4.0 gen) or `<version>` |
| `RATEL_SDK_VERSION` | `0.4.0` | `@ratel-ai/sdk` npm version pinned for the BFCL live-retrieval LLM eval |
| `RATEL_GENERATION` | `0.4.0` | `0.4.0` (SDK `--retriever`) \| `pre-0.4.0` (Rust retriever) |
| `RETRIEVER` | `bm25` | 0.4.0 gen only: `bm25` \| `semantic` \| `hybrid` |
| `RATEL_MODELS` | 3-link dict (below) | the models under test |
| `DOLLAR_GLOBAL` | `60` | hard USD cap per eval command |
| `REBASELINE_CONTROLS` | `false` | `true` once: run control arms live with `--force` (see below) |

**Not variables** (fixed experiment design): pool sizes, top-k, scenarios,
seed, arms. Changing those is a deliberate PR to `buildspec.yml` + EXPERIMENTS.md.

## Models — every entry is an endpoint link

`RATEL_MODELS` accepts a comma list or a JSON dict of links. Default:

```json
{"models": [
  "https://bedrock-runtime.eu-central-1.amazonaws.com/openai/v1#eu.anthropic.claude-sonnet-4-6",
  "https://bedrock-runtime.eu-central-1.amazonaws.com/openai/v1#eu.anthropic.claude-haiku-4-5-20251001-v1:0",
  "https://api.openai.com/v1#gpt-5.4-mini"
]}
```

Adding a model = appending one link. Link patterns:

| Provider | Link | Auth |
|---|---|---|
| Bedrock | `https://bedrock-runtime.<region>.amazonaws.com/openai/v1#<inference-profile-id>` | IAM role (no key) |
| Anthropic API | `https://api.anthropic.com/v1#<model>` | SSM `ANTHROPIC_API_KEY` |
| OpenAI | `https://api.openai.com/v1#<model>` | SSM `OPENAI_API_KEY` |
| Self-hosted (e.g. qwen3-4b via [ratel-inference-gateway](https://github.com/ratel-ai/ratel-inference-gateway)) | `https://<gateway>.execute-api.<region>.amazonaws.com/prod/v1#qwen3-4b` | SSM `MODEL_API_KEY` (bearer) |

Provider hosts route through their **native APIs** with the friendly model id
(`claude-sonnet-4-6`) stamped on every row — identical behavior/pricing/caching
to name-addressed models. Other hosts use the generic OpenAI-compatible client
and are auto-warmed (`POST /warm`).

Pricing: known models are priced automatically (Bedrock profile decoration is
stripped when resolving the `#fragment`); a genuinely new model runs fine but
records `dollar_cost=0` unless priced via the `RATEL_PRICING_JSON` env var —
which also means `DOLLAR_GLOBAL` can't see its spend, so price anything expensive.

A new model has no cached control arms: its first run executes all three arms
live (~3–4× its ratel-full cost). Subsequent runs reuse its controls from cache.

## Secrets (SSM Parameter Store — free tier, no Secrets Manager)

```bash
aws ssm put-parameter --profile ratel-bench \
  --name /ratel/bench/OPENAI_API_KEY --type SecureString --overwrite --value "sk-..."
```

| Parameter | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | yes | gpt-* arms |
| `RATEL_DEPLOY_KEY` | for `--tag`/`--rev` runs | read-only ed25519 deploy key for private `ratel-ai/ratel` (multi-line: use `--value file:///path/to/key`) |
| `ANTHROPIC_API_KEY` | optional | only for `api.anthropic.com` links / `RATEL_LLM_BACKEND=anthropic` |
| `MODEL_API_KEY` | optional | bearer for self-hosted endpoint links |

Placeholders (`PLACEHOLDER-set-via-put-parameter`) are skipped by the buildspec.
Claude-via-Bedrock needs **no secret** — the CodeBuild service role signs requests.

One-time account prerequisite (already done): Bedrock's Anthropic use-case form —
without it every Claude invoke 404s. Resubmit if ever needed:
`aws bedrock put-use-case-for-model-access --form-data '<json>'`
(fields: companyName, companyWebsite, intendedUsers:"0", industryOption,
otherIndustryOption, useCases).

## Results → publishing

The build syncs everything to S3; **publishing stays a git operation** (the
website rebuilds from `main`). After a successful run, review and push:

```bash
aws s3 sync s3://ratel-bench-results-340303437758/ results/ \
  --profile ratel-bench --exclude "test-data/*"
git add results/ && git commit -m "results: <version> benchmark run" && git push
```

## Bedrock control re-baseline (one-time, ~$106)

LLM-eval numbers for the Claude arms shift slightly on Bedrock vs the Anthropic
API (same models, different serving stack). Before publishing Bedrock-era
numbers, refresh the version-independent control arms once **on Bedrock**:

```bash
aws codebuild start-build --profile ratel-bench --region eu-central-1 \
  --project-name ratel-bench \
  --environment-variables-override name=REBASELINE_CONTROLS,value=true
```

That run passes `--force` so `control-baseline`/`control-oracle` execute live
and refresh the canonical `agent.jsonl` caches in S3. Every later run reuses
them (only `ratel-full` runs live, ≈$16/version).

## Toolchain image

The ECR image is toolchain-only (Rust, Node 24/pnpm, AWS CLI, baked
`BAAI/bge-small-en-v1.5` in `HF_HOME`) — **no repo code, no test-data**. Code
changes need no image work: every build clones the repo fresh. Rebuild only
when `Dockerfile` changes:

```bash
docker buildx build --platform linux/arm64 \
  -t 340303437758.dkr.ecr.eu-central-1.amazonaws.com/ratel-bench:latest --load .
aws ecr get-login-password --profile ratel-bench --region eu-central-1 |
  docker login --username AWS --password-stdin 340303437758.dkr.ecr.eu-central-1.amazonaws.com
docker push 340303437758.dkr.ecr.eu-central-1.amazonaws.com/ratel-bench:latest
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Claude invoke → 404 "use case details have not been submitted" | Anthropic use-case form missing (see Secrets section); allow ~15 min after submitting |
| `CreateProject` → `OAuthProviderException` | CodeBuild needs the GitHub connection imported as the account source credential: `aws codebuild import-source-credentials --server-type GITHUB --auth-type CODECONNECTIONS --token <connection-arn>` (an inline `auth{}` block in Terraform is rejected) |
| GitHub connection PENDING forever | The AWS Connector for GitHub app must be installed on the **ratel-ai org** (not a personal account) with access to this repo, then "Update pending connection" in the console — eu-central-1, watch for the console silently switching region |
| Build fails cloning `ratel-ai/ratel` | `RATEL_DEPLOY_KEY` placeholder/invalid, or the public half isn't a deploy key on that repo |
| Model runs but `dollar_cost` is 0 | No pricing entry — expected for self-hosted; for new cloud models set `RATEL_PRICING_JSON` |
| SSO errors (`Token has expired`) | `aws sso login --profile ratel-admin` |
