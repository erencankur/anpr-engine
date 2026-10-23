# Standalone runtime: no legacy base image, Torch, checkpoint or second venv.
FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS builder
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1
WORKDIR /build
RUN python -m pip install --no-cache-dir uv==0.12.5
COPY deploy/docker/requirements.lean-corners-amd64.txt /build/requirements.txt
RUN python -m venv /opt/paddle-venv && \
    uv pip sync --python /opt/paddle-venv/bin/python --require-hashes --only-binary :all: /build/requirements.txt
COPY deploy/docker/create_python_notices.py /build/create_python_notices.py
RUN python /build/create_python_notices.py /opt/paddle-venv /build/licenses

FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS runtime
ENV PATH=/opt/paddle-venv/bin:$PATH PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp XDG_CACHE_HOME=/tmp ANPR_RUNTIME_REPOSITORY_ROOT=/app ANPR_SERVICE_HOST=0.0.0.0 ANPR_SERVICE_PORT=8080 ANPR_INFERENCE_DEVICE=cpu ANPR_RUNTIME_PROFILE=lean_corners ANPR_WEB_OCR_CONFIG=/app/configs/web-ocr-learned-corners.json OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libgl1 libglib2.0-0 && \
    rm -r /var/lib/apt/lists/*
COPY --from=builder --chown=10001:10001 /opt/paddle-venv/ /opt/paddle-venv/
COPY --from=builder --chown=10001:10001 /build/licenses/ /licenses/python/
COPY --chown=10001:10001 src/anpr_engine/ /app/src/anpr_engine/
COPY --chown=10001:10001 runtimes/paddleocr-worker/ /app/runtimes/paddleocr-worker/
COPY --chown=10001:10001 runtimes/lean-smoke/smoke.py /app/runtimes/lean-smoke/smoke.py
COPY --chown=10001:10001 configs/public/web-ocr-learned-corners-template.json /app/configs/web-ocr-learned-corners.json
COPY --chown=10001:10001 scripts/launch_private_inference_service.py /app/scripts/launch_private_inference_service.py
RUN mkdir -p /licenses/base && dpkg-query -W -f='${Package}\t${Version}\n' > /licenses/base/debian-packages.tsv && \
    find /usr/share/doc -name copyright | while read -r file; do cp --parents "$file" /licenses/base/; done
USER 10001:10001
RUN python -c 'import importlib.util; assert importlib.util.find_spec("torch") is None; from anpr_engine.inference_service.app import create_app; from anpr_engine.integration.lean_corners import LeanCornerPipeline'
EXPOSE 8080
CMD ["python", "/app/scripts/launch_private_inference_service.py"]
