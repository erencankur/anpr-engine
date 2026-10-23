# Build the current-source inference.amd64.Dockerfile first. This stage adds
# a separate OCR environment; it does not embed any private model weights.
ARG ANPR_INFERENCE_BASE_IMAGE=anpr-engine-inference:web-cpu-base
FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS paddle-builder
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_BUILDS=1
WORKDIR /build
RUN python -m pip install --no-cache-dir uv==0.12.5
COPY deploy/docker/requirements.paddle-native-amd64.txt /build/requirements.txt
RUN python -m venv /opt/paddle-venv && \
    uv pip sync --python /opt/paddle-venv/bin/python --require-hashes --only-binary :all: /build/requirements.txt
COPY deploy/docker/create_python_notices.py /build/create_python_notices.py
RUN python /build/create_python_notices.py /opt/paddle-venv /build/licenses/paddleocr

FROM ${ANPR_INFERENCE_BASE_IMAGE} AS runtime
USER 0:0
# Native CPU inference runtime; OpenCV is installed as a headless wheel.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libgl1 libglib2.0-0 && \
    rm -r /var/lib/apt/lists/*
COPY --from=paddle-builder --chown=10001:10001 /opt/paddle-venv/ /opt/paddle-venv/
COPY --from=paddle-builder --chown=10001:10001 /build/licenses/paddleocr/ /licenses/paddleocr/
COPY --chown=10001:10001 runtimes/paddleocr-worker/worker.py /app/runtimes/paddleocr-worker/worker.py
COPY --chown=10001:10001 runtimes/paddleocr-worker/native_backend.py /app/runtimes/paddleocr-worker/native_backend.py
COPY --chown=10001:10001 configs/public/web-ocr-runtime-template.json /app/configs/web-ocr-runtime.json
COPY --chown=10001:10001 configs/public/web-ocr-learned-corners-template.json /app/configs/web-ocr-learned-corners.json
RUN dpkg-query -W -f='${Package}\t${Version}\n' > /licenses/base/debian-packages.tsv && \
    find /usr/share/doc -name copyright | while read -r file; do cp --parents "$file" /licenses/base/; done
ENV ANPR_WEB_OCR_CONFIG=/app/configs/web-ocr-runtime.json PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
USER 10001:10001
# Check dependency imports without downloading or initializing a model.
RUN /opt/paddle-venv/bin/python -c 'import sys; sys.path.insert(0, "/app/runtimes/paddleocr-worker"); from native_backend import NativeCtcRecognizer; from paddle import inference'
