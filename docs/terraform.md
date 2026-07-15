# Terraform on AWS

## Provisioning EC2

Terraform declares infrastructure as code. An `aws_instance` resource describes
the desired EC2 instance; `terraform plan` shows the diff between declared and
actual state, and `terraform apply` reconciles them.

## State management

Terraform tracks real resources in a state file. Local state does not work for
teams: two engineers applying at once will corrupt it. Remote state in an S3
bucket with a DynamoDB table for state locking is the standard pattern.

## Security groups

Security groups are stateful firewalls attached to instances. Return traffic for
an allowed inbound connection is automatically permitted, so egress rules do not
need a matching entry.
