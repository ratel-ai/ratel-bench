variable "region" {
  description = "Workload region (Bedrock/Claude coverage + EU residency)."
  type        = string
  default     = "eu-central-1"
}

variable "profile" {
  description = "AWS CLI profile for the backend/caller. null = default chain (e.g. AWS_PROFILE). Use ratel-admin locally."
  type        = string
  default     = null
}

variable "benchmarks_account_id" {
  description = "benchmarks member account id (from ratel-infra org/)."
  type        = string
  default     = "340303437758"
}

variable "project_name" {
  description = "Name prefix for CodeBuild / ECR / log group."
  type        = string
  default     = "ratel-bench"
}

variable "github_repo_url" {
  description = "HTTPS clone URL CodeBuild pulls code + buildspec.yml from."
  type        = string
  default     = "https://github.com/ratel-ai/ratel-bench.git"
}

variable "results_bucket_name" {
  description = "Versioned S3 bucket mirroring results/ (raw cells, caches, reports) + test-data/."
  type        = string
  default     = "ratel-bench-results-340303437758"
}

variable "ecr_repo_name" {
  description = "ECR repo for the slim toolchain image."
  type        = string
  default     = "ratel-bench"
}

variable "ssm_prefix" {
  description = "SSM Parameter Store root for the SecureString .env values (free tier; no Secrets Manager)."
  type        = string
  default     = "/ratel/bench"
}

variable "default_models" {
  description = "Default RATEL_MODELS for a run: endpoint links (Claude via Bedrock + gpt via OpenAI). Add a model by appending its link — see the buildspec header."
  type        = string
  default     = "{\"models\":[\"https://bedrock-runtime.eu-central-1.amazonaws.com/openai/v1#eu.anthropic.claude-sonnet-4-6\",\"https://bedrock-runtime.eu-central-1.amazonaws.com/openai/v1#eu.anthropic.claude-haiku-4-5-20251001-v1:0\",\"https://api.openai.com/v1#gpt-5.4-mini\"]}"
}

variable "dollar_global_default" {
  description = "Default hard USD cap per run."
  type        = string
  default     = "60"
}
