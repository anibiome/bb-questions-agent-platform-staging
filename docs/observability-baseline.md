# Observability Baseline

## Logging

- Emit structured logs with stable event names.
- Include request id, user id (or anonymized id), endpoint, and duration.
- Never log raw sensitive payloads.
- Use a stable error taxonomy field (`error_type`) for all non-2xx responses.

## Metrics

Track these minimum service-level metrics:

- request rate and latency (`p50`, `p95`, `p99`),
- error rate by endpoint and status class,
- policy decision throughput,
- snapshot write/read failures.

## Error Taxonomy

Use these top-level classes in logs and alerts:

- `validation_error`: malformed request/schema violations.
- `policy_error`: policy runtime selection or rollback guard failures.
- `storage_error`: DB read/write/migration failures.
- `integration_error`: external contract or adapter failures.
- `internal_error`: unexpected exceptions.

## Alerts

Minimum alert conditions:

- sustained 5xx error rate above threshold,
- latency breach on core endpoints,
- migration failures,
- contract validation failures in CI.

Recommended initial thresholds:

- `5xx_rate > 2%` for 10 minutes.
- `p95_latency > 800ms` for 15 minutes on `/v1/users/*/daily-questions` and `/v1/users/*/answers`.
- `snapshot_write_failure_rate > 1%` over rolling 14 days.

## Runtime Controls

- Always keep health endpoint live.
- Fail startup on migration incompatibility.
- Keep audit trail around policy decisions and outcomes.
