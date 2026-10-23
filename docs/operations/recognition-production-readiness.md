# Recognition production-readiness gate

## Current outcome

The no-GPU Recognition engineering scope is complete, but production promotion is correctly
`BLOCKED`. This is not a software failure: final acceptance evidence does not exist and must not be
fabricated from reused development data.

The current readiness input identity is
`b8efc42a429d07e6621daaa39863bf6d1a93edbd6e2bcfb91585be0e0c7b3890`. The immutable aggregate
report SHA-256 is `9e3b11ea5debe12084a3968cfb80e54a718f29a3f3a40d948f24c0ee390a24aa`.
It records `engineering_complete=true`, `activation_performed=false`, and
`production_route_changed=false`.

## Fail-closed blocker register

| Blocker | Required resolution |
| --- | --- |
| `EVALUATION_PROTOCOL_NOT_FROZEN` | Choose numeric product-risk gates and bind the target hardware before final-data access |
| `INDEPENDENT_HOLDOUT_NOT_ADMITTED` | Obtain authorized, unused, group-isolated evidence and pass the admission audit |
| `NEAR_DUPLICATE_REVIEW_INCOMPLETE` | Complete private perceptual/source-lineage review after holdout admission |
| `FINAL_PAIRED_EVALUATION_MISSING` | Run baseline and candidate once on the sealed holdout |
| `FINAL_GATES_NOT_PASSED` | Verify every predeclared metric and critical slice |
| `TARGET_HOST_PROFILE_MISSING` | Identify the real deployment host and record its immutable profile |
| `TARGET_HOST_LIMITS_NOT_PASSED` | Pass cold-start, latency, memory, and failure limits on that host |
| `LIVE_SHADOW_NOT_COMPLETED` | Complete a controlled observation window after final acceptance |
| `TARGET_HOST_ROLLBACK_NOT_COMPLETED` | Prove candidate stop and baseline restoration on the deployment host |
| `OWNER_ACTIVATION_NOT_APPROVED` | Obtain explicit approval only after all preceding gates pass |

CUDA, RTMDet, learned-corner training, full PNG export, and the deferred 229 corner reviews are not
blockers in this active path.

## Re-run

Update the private evidence state only when genuine new evidence exists, then run:

```text
PYTHONPATH=src .venv/bin/python scripts/audit_recognition_promotion_readiness.py \
  --input configs/public/recognition-promotion-readiness-template.yaml \
  --output <new-non-existing-report-path>.json
```

The auditor never activates a route. Even a complete input can yield only
`ELIGIBLE_FOR_EXPLICIT_ACTIVATION`; `activation_performed` remains false. Existing reports cannot be
overwritten.
