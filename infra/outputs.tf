output "ecr_repository_url" {
  description = "Push the slim toolchain image here (:latest)."
  value       = aws_ecr_repository.image.repository_url
}

output "results_bucket" {
  description = "S3 bucket for raw cells, caches, reports, and test-data/."
  value       = aws_s3_bucket.results.bucket
}

output "codebuild_project" {
  description = "Start a run: aws codebuild start-build --project-name <this>."
  value       = aws_codebuild_project.bench.name
}

output "codebuild_role_arn" {
  value = aws_iam_role.codebuild.arn
}

output "github_connection_arn" {
  description = "AUTHORIZE ONCE in the console (Developer Tools → Connections) — starts PENDING."
  value       = aws_codestarconnections_connection.github.arn
}

output "log_group" {
  value = aws_cloudwatch_log_group.build.name
}
