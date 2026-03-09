# Security Policy

## Supported Versions

Security fixes are applied on the `main` branch.

## Reporting a Vulnerability

Please do not open public issues for security vulnerabilities.

Report privately with:
- Subject: `Security report: questions-agent-platform`
- Affected area (API endpoint, migration, policy runtime, data handling, etc.)
- Reproduction steps and expected impact
- Proof-of-concept or logs if available

Until a dedicated security mailbox is configured, contact the repository maintainer directly through private GitHub communication.

## Response Targets

- Acknowledgement: within 48 hours
- Initial triage: within 5 business days
- Fix plan or mitigation guidance: within 10 business days

## Scope

In-scope areas include:
- API auth and authorization behavior
- Data exposure across user boundaries
- Input validation and injection risks
- Production migration safety
- Backup/restore and operational tooling

## Operational Guardrails

- Default ANI patient data, proprietary datasets, prompts, weights, methods, and model outputs to private access.
- Do not expose Cloud Run services, buckets, object links, or dashboards to anonymous internet access by default.
- Do not send patient data, ANI runtime context, or proprietary IP directly from browser/mobile clients to third-party LLM endpoints. Route through ANI-controlled backend APIs.
- Do not commit plaintext secrets, database credentials, or admin keys. Use server-side secret management and least-privilege service accounts.
- Do not script or document broad project IAM grants without explicit owner approval and security review.
- Do not delete, purge, or migrate production/study data without explicit approval, a backup/snapshot path, and a rollback plan.
- Before deploy or handoff from the main ANI workspace, run `./workspace_governance/workspace_security_redline` and treat findings as blockers.
