data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.name
}

# ─────────────────────────────── ECR ───────────────────────────────
# Slim toolchain image (Rust + Node/pnpm + AWS CLI + baked bge-small).
resource "aws_ecr_repository" "image" {
  name                 = var.ecr_repo_name
  image_tag_mutability = "MUTABLE" # :latest is re-pushed on toolchain/model bumps

  image_scanning_configuration {
    scan_on_push = true
  }
}

# ───────────────────────────── S3 (state of results) ─────────────────────────────
# Mirrors results/ (raw cells + caches + reports) and hosts test-data/.
resource "aws_s3_bucket" "results" {
  bucket = var.results_bucket_name
}

resource "aws_s3_bucket_versioning" "results" {
  bucket = aws_s3_bucket.results.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "results" {
  bucket = aws_s3_bucket.results.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "results" {
  bucket                  = aws_s3_bucket.results.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ──────────────────────── GitHub source connection ────────────────────────
# CodeBuild pulls the repo + buildspec via CodeConnections. Created here, but the
# first apply leaves it PENDING — authorize it ONCE in the console
# (Developer Tools → Settings → Connections → Update pending connection).
resource "aws_codestarconnections_connection" "github" {
  name          = "${var.project_name}-github"
  provider_type = "GitHub"
}

# ──────────────────────────── Logs ────────────────────────────
resource "aws_cloudwatch_log_group" "build" {
  name              = "/aws/codebuild/${var.project_name}"
  retention_in_days = 30
}

# ──────────────────────────── IAM (CodeBuild service role) ────────────────────────────
resource "aws_iam_role" "codebuild" {
  name = "${var.project_name}-codebuild"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "codebuild.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "codebuild" {
  name = "${var.project_name}-codebuild"
  role = aws_iam_role.codebuild.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "Logs"
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = ["${aws_cloudwatch_log_group.build.arn}:*"]
      },
      {
        Sid      = "EcrAuth"
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = ["*"]
      },
      {
        Sid      = "EcrPull"
        Effect   = "Allow"
        Action   = ["ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"]
        Resource = [aws_ecr_repository.image.arn]
      },
      {
        Sid      = "ResultsBucket"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
        Resource = [aws_s3_bucket.results.arn, "${aws_s3_bucket.results.arn}/*"]
      },
      {
        Sid      = "SsmEnv"
        Effect   = "Allow"
        Action   = ["ssm:GetParameter", "ssm:GetParameters"]
        Resource = ["arn:aws:ssm:${local.region}:${local.account_id}:parameter${var.ssm_prefix}/*"]
      },
      {
        Sid      = "SsmDecrypt"
        Effect   = "Allow"
        Action   = ["kms:Decrypt"]
        Resource = ["*"]
        Condition = {
          StringEquals = { "kms:ViaService" = "ssm.${local.region}.amazonaws.com" }
        }
      },
      {
        # Bedrock Claude via inference profiles (this account) + the underlying
        # anthropic foundation models across regions. Scoped to anthropic; tighten
        # to the exact profile ARNs once `aws bedrock list-inference-profiles` is run.
        Sid    = "BedrockClaude"
        Effect = "Allow"
        Action = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
        Resource = [
          "arn:aws:bedrock:*:${local.account_id}:inference-profile/*",
          "arn:aws:bedrock:*::foundation-model/anthropic.*"
        ]
      },
      {
        Sid      = "UseGithubConnection"
        Effect   = "Allow"
        Action   = ["codestar-connections:UseConnection", "codeconnections:UseConnection"]
        Resource = [aws_codestarconnections_connection.github.arn]
      }
    ]
  })
}

# ──────────────────────────── CodeBuild project ────────────────────────────
resource "aws_codebuild_project" "bench" {
  name          = var.project_name
  description   = "ratel-bench benchmark pipeline (one build == one Ratel version)."
  service_role  = aws_iam_role.codebuild.arn
  build_timeout = 480 # minutes (8h)

  source {
    type            = "GITHUB"
    location        = var.github_repo_url
    git_clone_depth = 1
    buildspec       = "buildspec.yml"
    # Auth comes from the account-level CodeBuild source credential, which is the
    # CodeConnections connection imported via:
    #   aws codebuild import-source-credentials --server-type GITHUB \
    #     --auth-type CODECONNECTIONS --token <connection-arn>
    # (An inline auth{} block here is rejected with OAuthProviderException.)
  }

  source_version = "RAT-374/aws-architecture" # default branch to build; override per run

  environment {
    type                        = "ARM_CONTAINER"
    compute_type                = "BUILD_GENERAL1_SMALL"
    image                       = "${aws_ecr_repository.image.repository_url}:latest"
    image_pull_credentials_type = "SERVICE_ROLE"

    environment_variable {
      name  = "RESULTS_BUCKET"
      value = aws_s3_bucket.results.bucket
    }
    environment_variable {
      name  = "SSM_PREFIX"
      value = var.ssm_prefix
    }
    environment_variable {
      name  = "AWS_REGION"
      value = var.region
    }
    environment_variable {
      name  = "RATEL_LLM_BACKEND"
      value = "bedrock"
    }
    environment_variable {
      name  = "RATEL_MODELS"
      value = var.default_models
    }
    environment_variable {
      name  = "DOLLAR_GLOBAL"
      value = var.dollar_global_default
    }

    # Per-run knobs, surfaced at project level so the console's "Start build
    # with overrides" page shows them pre-filled (buildspec env.variables are
    # invisible there). Values mirror the buildspec defaults = latest release.
    environment_variable {
      name  = "RETRIEVER"
      value = "bm25" # 0.4.0 generation only: bm25 | semantic | hybrid
    }
    environment_variable {
      name  = "RATEL_VERSION_ARGS"
      value = "--crate 0.4.0"
    }
    environment_variable {
      name  = "RATEL_EXPECT"
      value = "0.4.0"
    }
    environment_variable {
      name  = "RATEL_SDK_VERSION"
      value = "0.4.0"
    }
    environment_variable {
      name  = "RATEL_GENERATION"
      value = "0.4.0" # "pre-0.4.0" for versions before the SDK --retriever era
    }
    environment_variable {
      name  = "RATEL_VERSION_LABEL"
      value = "auto" # "auto" → derived from RATEL_EXPECT + RETRIEVER in the buildspec
    }
    environment_variable {
      name  = "REBASELINE_CONTROLS"
      value = "false" # "true" once: re-baseline control arms live on Bedrock (--force)
    }
    environment_variable {
      name  = "RUN_STAGE"
      value = "both" # both | retrieval (free, no LLM) | llm (reuse S3 retrieval rows)
    }
  }

  cache {
    type  = "LOCAL"
    modes = ["LOCAL_CUSTOM_CACHE", "LOCAL_SOURCE_CACHE"]
  }

  artifacts {
    type = "NO_ARTIFACTS" # results (incl. report.json) are pushed to S3 by the buildspec
  }

  logs_config {
    cloudwatch_logs {
      group_name = aws_cloudwatch_log_group.build.name
    }
  }
}

# ──────────────────────── SSM SecureString .env values (free tier) ────────────────────────
# TF owns the parameter resources; the SECRET VALUES are populated out-of-band
# (ROAM-8: `aws ssm put-parameter --overwrite`), so no secret ever lands in tfstate.
resource "aws_ssm_parameter" "openai_api_key" {
  name  = "${var.ssm_prefix}/OPENAI_API_KEY"
  type  = "SecureString"
  value = "PLACEHOLDER-set-via-put-parameter"
  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "ratel_deploy_key" {
  name  = "${var.ssm_prefix}/RATEL_DEPLOY_KEY"
  type  = "SecureString"
  value = "PLACEHOLDER-set-via-put-parameter"
  lifecycle {
    ignore_changes = [value]
  }
}

# Optional: only needed for runs with RATEL_LLM_BACKEND=anthropic (claude-* via
# the Anthropic API instead of Bedrock). The buildspec skips it while it still
# holds the placeholder.
resource "aws_ssm_parameter" "anthropic_api_key" {
  name  = "${var.ssm_prefix}/ANTHROPIC_API_KEY"
  type  = "SecureString"
  value = "PLACEHOLDER-set-via-put-parameter"
  lifecycle {
    ignore_changes = [value]
  }
}

# Optional: bearer token for endpoint-link models in RATEL_MODELS
# (`<baseURL>#<model>`), e.g. a Bedrock API key. The buildspec exports it as
# AWS_BEDROCK_BEARER; skipped while it still holds the placeholder.
resource "aws_ssm_parameter" "model_api_key" {
  name  = "${var.ssm_prefix}/MODEL_API_KEY"
  type  = "SecureString"
  value = "PLACEHOLDER-set-via-put-parameter"
  lifecycle {
    ignore_changes = [value]
  }
}
