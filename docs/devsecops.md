# DevSecOps Practices

## Secret leakage

A credential committed to Git is compromised the moment it is pushed, even if
the commit is later reverted. Reverting removes the file from HEAD but leaves it
in history, still retrievable via the reflog or by any existing clone.

Remediation has three parts, in order:
1. Rotate the credential immediately. This is the only step that actually
   restores security; the rest is cleanup.
2. Rewrite history with `git filter-repo` to purge the blob, then force-push.
3. Add pre-commit secret scanning so it cannot recur.

## OWASP Top 10

The OWASP Top 10 catalogues the most critical web application security risks:
broken access control, cryptographic failures, injection, insecure design,
security misconfiguration, vulnerable components, authentication failures,
software and data integrity failures, logging and monitoring failures, and
server-side request forgery.

## Least privilege

An IAM role should grant the narrowest permission set that lets the workload do
its job. Wildcards in IAM policies are how a compromised container becomes a
compromised account.
