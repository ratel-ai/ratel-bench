# Version pins — match ratel-infra conventions (terraform >= 1.10, aws ~> 5.0).
terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}
