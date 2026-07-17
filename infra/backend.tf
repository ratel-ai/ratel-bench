# Remote state in the shared ratel-infra bootstrap bucket (management account),
# one key per config, native S3 locking — same pattern as every ratel-infra config.
terraform {
  backend "s3" {
    bucket       = "ratel-tfstate-288742312794"
    key          = "ratel-bench/terraform.tfstate"
    region       = "eu-central-1"
    encrypt      = true
    use_lockfile = true
  }
}
