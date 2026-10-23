# Isolated PaddleOCR Recognition worker

## Scope and current state

This is an evaluation-only boundary for the selected PaddleOCR Recognition candidate. It keeps
PaddleOCR, PaddlePaddle, and their OpenCV distribution outside the main ANPR environment. It does
not alter the active project-owned inference route and is not a production activation approval.

The repository contains:

- a strict in-process request/response contract and provider-independent recognizer adapter;
- a bounded, single-flight JSON-lines subprocess transport;
- a standalone private worker under `runtimes/paddleocr-worker/worker.py`; and
- worker-only pinned direct requirements.

The worker accepts encoded PNG bytes, never a host file path. It verifies the byte count, SHA-256,
decodability, pixel count, and exact request fields. At startup it also verifies the exact local
model-directory SHA-256 before importing PaddleOCR. Provider errors are returned as finite safe
codes; plate text is returned only on the private protocol and must never be written to logs.

## Why the worker is separate

The main service pins `opencv-python-headless`, while the tested PaddleOCR dependency tree brings
its own `opencv-contrib-python` distribution. Both provide the `cv2` namespace. Installing them in
one environment can silently change runtime behavior, so co-installation is prohibited.

Official PaddleOCR guidance separates inference-engine installation from the package and
recommends a clean environment when dependency conflicts are possible:

- <https://www.paddleocr.ai/main/en/version3.x/installation.html>
- <https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html>

## Recorded candidate environment

The controlled development benchmark has now been completed with:

- PaddleOCR 3.7.0 and PaddlePaddle 3.3.1 in a dedicated Python 3.11 inference environment;
- OpenCV contrib 4.10.0 only in that environment, leaving the main service environment intact;
- official `latin_PP-OCRv5_mobile_rec` inference-directory SHA-256
  `ce3f608dd613dafc8329e3b781453e790deaa19a43445e36c682cc10e1422215`;
- official model-card `license: apache-2.0` evidence SHA-256
  `897e932f55386124163c5d051ee42a7481fad52ea447c9e3b009d22e5424bada`;
- a separate training environment and PaddleOCR source checkout pinned to revision
  `cdc6d02f680f922b2f8c3605f716d3022a9c9a0f`; and
- a project fine-tuned inference artifact with directory SHA-256
  `5fffa451df2efa707bafd3d1da7e94aaebf83c5d365d2c4b8a6d439b4ac48964`.

The environments and all model artifacts remain outside the public repository. The full result is
documented in
[`../reports/paddleocr-recognition-development-validation.md`](../reports/paddleocr-recognition-development-validation.md).
No Docker image was required, and no full PNG Detection dataset was created.

## Controlled local invocation

Create a dedicated virtual environment outside the main `.venv`, install
`runtimes/paddleocr-worker/requirements.txt`, and launch the worker with the reviewed local model
directory and its exact SHA-256:

```text
<isolated-python> runtimes/paddleocr-worker/worker.py \
  --model-dir <reviewed-model-directory> \
  --model-sha256 <64-lowercase-hex-digest> \
  --device cpu \
  --cpu-threads 2 \
  --engine paddle_static
```

The parent process must keep stdin/stdout private pipes, send one request at a time, enforce its
timeout, and terminate the worker on timeout, malformed output, request-ID mismatch, or broken
pipe. The provided transport implements these rules. The current public launcher does not expose
this command or activate this worker.

## Resource-safe local policy

The worker defaults to two CPU threads and applies the same bound to PaddleX, OpenMP, OpenBLAS,
MKL, Accelerate/vecLib, and NumExpr before importing the provider runtime. The parent transport is
single-flight, so one worker cannot process requests concurrently. Local development must keep the
candidate disabled by default, start it only for a bounded command, and close it immediately when
the command finishes.

A three-request resource smoke on the 16 GiB development Mac observed a maximum sampled worker RSS
of 865.95 MiB after warm-up. This is a sampled resident-memory observation, not a guaranteed peak.
The worker exited after the command and no background process remained. Long soak tests should not
be repeated unless runtime changes require them.

## Disabled-by-default shadow and fallback boundary

`RecognitionShadowConfig` accepts only `DISABLED` or `SHADOW`; there is intentionally no active or
production-selection mode. `DISABLED` never invokes the candidate factory. In `SHADOW`, the
project Recognition result remains authoritative even when the candidate succeeds. A candidate
startup, timeout, inference, protocol, or cleanup failure cannot replace that result.

The first candidate failure opens a circuit by default, closes any worker, and prevents repeated
restart attempts. Retrying requires an explicit operator reset. A worker is also recycled after a
bounded number of requests. The public template remains disabled at
`configs/public/recognition-shadow-template.yaml`; loading a template does not connect it to the
active v1 service.

The controller exposes aggregate counters for requests, disabled decisions, candidate successes,
candidate failures, circuit-open skips, session starts/closes, cleanup failures, and explicit
resets. The snapshot contains no crop, path, image identity, plate text, prediction, confidence, or
exception message. It is safe for operational health reporting but is not an accuracy report.

The development health snapshot reports only `DISABLED`, `READY`, or `CIRCUIT_OPEN`, binds the
effective strict configuration by canonical SHA-256, and permanently declares the baseline as the
authoritative source. Calling `health()` does not create a candidate session or start a process.
The snapshot also declares `candidate_can_be_authoritative=false` and
`production_promotion_eligible=false`.

## Control-plane rehearsal

The synthetic rehearsal command exercises disabled/no-start, healthy shadow, bounded worker
recycling, candidate failure fallback, circuit-open restart suppression, explicit reset, recovery,
and final close without importing or loading PaddleOCR:

```text
PYTHONPATH=src .venv/bin/python scripts/rehearse_recognition_shadow_control.py \
  --output <new-report-path>.json
```

The retained 2026-10-14 rehearsal passed all seven checks. Its aggregate report SHA-256 is
`8cd617e5086b5f88110b3e482fafedb849f8c8a95ae1801d1e54484e9f575232`. The report declares that
it used no real model or private data, contains no inference content, and changed no production
route. Existing output paths are never overwritten.

## Validation completed

- RGB-to-provider-BGR byte parity through PNG encode/decode;
- image byte/hash/decode bounds and rejection behavior;
- strict request and response fields;
- safe provider/runtime error mapping;
- multiple messages through a persistent subprocess;
- two-thread native CPU-pool limits and explicit provider close on worker shutdown;
- disabled-mode no-start behavior, baseline-authoritative shadow behavior, fail-safe fallback,
  circuit opening/reset, bounded worker recycling, and lazy managed-session lifecycle;
- privacy-safe aggregate lifecycle counters and cleanup-failure accounting with no plate content;
- side-effect-free development health snapshots and deterministic shadow-configuration identity;
- deterministic synthetic control-plane rehearsal with immutable aggregate output;
- fail-closed malformed subprocess output and response-ID mismatch;
- equal artifact-directory identity between the parent and worker implementations.
- native Paddle inference on all 221 governed validation crops;
- zero-shot and fine-tuned aggregate evaluation;
- paired comparison against the frozen project recognizer;
- descriptive confidence/coverage curves with no retained plate text or sample identifiers; and
- one-epoch fine-tuning plus inference export in a separate pinned training environment.

The development candidate reached 90.95% exact normalized plate accuracy and 1.68% character error
rate on the 221-record development validation split. These numbers are not production acceptance:
an independent final holdout, product-approved operating threshold, detector/rectified crop track,
deployment-backend parity, and target-host performance remain pending.
