# semaje Whisper STT service — faster-whisper backend.
# (whisper.cpp is a host-native optimization; containers use faster-whisper.)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY services/whisper/requirements.txt services/whisper/requirements.txt
RUN pip install --no-cache-dir -r services/whisper/requirements.txt

COPY services/whisper/ services/whisper/
RUN touch services/__init__.py

ENV WHISPER_BACKEND=faster-whisper \
    WHISPER_MODEL=base \
    WHISPER_PORT=8011 \
    HF_HOME=/models

VOLUME /models
EXPOSE 8011
CMD ["python", "-m", "services.whisper.app"]
