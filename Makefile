# semaje-stt — common tasks. The venv lives outside the repo (see setup.sh).
SEMAJE_HOME ?= $(HOME)/.local/share/semaje
PY ?= $(SEMAJE_HOME)/venv-stt/bin/python

.PHONY: setup run test eval eval-fleurs image image-cuda
setup:        ## venv + models
	./setup.sh
run:          ## serve on :8011
	./run.sh
test:         ## gate tests (no models, <5 s; needs ffmpeg)
	$(PY) -m pytest -q
eval:         ## English smoke eval (macOS `say`)
	$(PY) -m semaje_stt.evals.eval_transcription
eval-fleurs:  ## real-speech WER/CER per language against a running service
	$(PY) -u -m semaje_stt.evals.eval_fleurs $(ARGS)
image:        ## CPU image (both engines on x86-64)
	docker build -t semaje-stt .
image-cuda:   ## NVIDIA image
	docker build -f Dockerfile.cuda -t semaje-stt:cuda .
