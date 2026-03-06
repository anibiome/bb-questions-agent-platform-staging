# Software Requirements

This document captures the current software-control baseline for `questions-agent-platform`.

| ID | Requirement |
| --- | --- |
| QAP-SWR-001 | The platform shall provide traceable daily-question and follow-up selection behavior through version-controlled policy and pipeline logic. |
| QAP-SWR-002 | Changes affecting scoring, safety events, projections, or state snapshots shall be regression-tested before release. |
| QAP-SWR-003 | Production authentication settings shall be supplied through controlled environment configuration. |
| QAP-SWR-004 | Rollout-gate and offline-evaluation logic shall remain reproducible from source-controlled code and inputs. |
| QAP-SWR-005 | Contract changes affecting downstream ANI services shall be versioned or explicitly governed. |
| QAP-SWR-006 | Committed fixtures and demo data shall remain non-production and de-identified. |
| QAP-SWR-007 | Placeholder production values in example configuration files shall not be used as deployed settings. |
| QAP-SWR-008 | The repository shall maintain code-owner review, CI execution, and dependency monitoring. |
| QAP-SWR-009 | The repository shall not be used as the authoritative store for patient-identifying production records. |
| QAP-SWR-010 | Release promotion shall capture commit state, environment, and configuration approval outside this repository. |
