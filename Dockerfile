# syntax=docker/dockerfile:1
# semaje-stt — Whisper (faster-whisper) + Omnilingual ASR on CPU.
#   docker build .                              # both engines (x86-64)
#   docker build --build-arg OMNILINGUAL=0 .     # Whisper only (~1.5 GB smaller)
#
# fairseq2 (Omnilingual's runtime) publishes Linux wheels for x86-64 only, on Meta's
# index, per torch version; on arm64 the image is Whisper-only and says so. kenlm (an
# Omnilingual dependency) has no wheels at all, so a builder stage compiles every wheel
# and the runtime image ships without a compiler.

ARG PYTHON_IMAGE=python:3.11-slim

FROM ${PYTHON_IMAGE} AS wheels
ARG OMNILINGUAL=1
ARG TARGETARCH
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake zlib1g-dev libbz2-dev liblzma-dev \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements*.txt ./
# One resolution per build so /wheels is always a consistent set.
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$OMNILINGUAL" = "1" ] && [ "$TARGETARCH" = "amd64" ]; then \
       pip wheel -w /wheels \
         --extra-index-url https://download.pytorch.org/whl/cpu \
         --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cpu \
         -r requirements.txt -r requirements-omnilingual.txt; \
     else \
       if [ "$OMNILINGUAL" = "1" ]; then \
         echo "WARNING: Omnilingual ASR has no prebuilt Linux wheels for $TARGETARCH; building Whisper-only." \
              "Build with --platform linux/amd64, or run ./setup.sh natively (Apple Silicon is supported natively)."; \
       fi; \
       pip wheel -w /wheels -r requirements.txt; \
     fi

FROM ${PYTHON_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Install exactly what the builder produced (Whisper-only or both engines); no network.
RUN --mount=type=bind,from=wheels,source=/wheels,target=/wheels \
    pip install --no-cache-dir --no-index --find-links /wheels /wheels/*.whl

COPY semaje_stt/ semaje_stt/
RUN useradd --system --uid 10001 semaje && mkdir -p /models && chown semaje /models
USER semaje

ENV STT_WHISPER_MODEL=base \
    STT_OMNILINGUAL_MODEL=omniASR_CTC_300M_v2 \
    STT_PORT=8011 \
    STT_MODEL_DIR=/models \
    HF_HOME=/models/hf \
    STT_WARMUP=true \
    PYTHONUNBUFFERED=1

VOLUME /models
EXPOSE 8011
HEALTHCHECK --interval=15s --timeout=5s --start-period=600s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8011/health/ready', timeout=3).status == 200 else 1)"
CMD ["python", "-m", "semaje_stt.app"]
