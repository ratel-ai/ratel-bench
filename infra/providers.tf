# Workload resources live in the benchmarks member account. The backend uses the
# caller's creds (ratel-admin, management account); the provider assumes into
# benchmarks via OrganizationAccountAccessRole — the same shape as
# ratel-infra/github-oidc/providers.tf.
provider "aws" {
  region  = var.region
  profile = var.profile

  assume_role {
    role_arn = "arn:aws:iam::${var.benchmarks_account_id}:role/OrganizationAccountAccessRole"
  }

  default_tags {
    tags = {
      ManagedBy = "terraform"
      Repo      = "ratel-bench"
      Config    = "infra"
    }
  }
}
