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
  description = "Default RATEL_MODELS for a run (Claude via Bedrock + gpt via OpenAI)."
  type        = string
  default     = "claude-sonnet-4-6,claude-haiku-4-5,gpt-5.4-mini"
}

variable "dollar_global_default" {
  description = "Default hard USD cap per run."
  type        = string
  default     = "60"
}
