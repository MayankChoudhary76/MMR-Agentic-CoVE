PYTHON := .venv/bin/python
PIP    := .venv/bin/pip
UVICORN:= .venv/bin/uvicorn

DATASET   ?= beauty
FUSION    ?= concat
W_TEXT    ?= 1.0
W_IMAGE   ?= 1.0
W_META    ?= 0.0
FAISS_NAME?= $(DATASET)_$(FUSION)_best
API_HOST  ?= 0.0.0.0
API_PORT  ?= 8000

export PYTHONPATH := $(shell pwd)

.PHONY: help
help:
	@echo "make install   # venv + deps"
	@echo "make prepare   # join + embeddings"
	@echo "make index     # build FAISS (vars: W_META, FAISS_NAME, ...)"
	@echo "make api       # run FastAPI"
	@echo "make ui        # run Gradio (needs API_URL env)"
	@echo "make test-api  # curl a recommendation"

.PHONY: venv
venv:
	python -m venv .venv

.PHONY: install
install: venv
	$(PIP) install -U pip wheel
	@if [ -f requirements.txt ]; then $(PIP) install -r requirements.txt; fi
	@if [ -f requirements-api.txt ]; then $(PIP) install -r requirements-api.txt; fi
	@if [ -f requirements-ui.txt ]; then $(PIP) install -r requirements-ui.txt; fi

.PHONY: prepare
prepare:
	$(PYTHON) -m scripts.run_agent --intent prepare --dataset $(DATASET)

.PHONY: index
index:
	$(PYTHON) -m scripts.run_agent --intent index \
		--dataset $(DATASET) --fusion $(FUSION) \
		--w_text $(W_TEXT) --w_image $(W_IMAGE) --w_meta $(W_META) \
		--faiss_name $(FAISS_NAME)

.PHONY: api
api:
	$(UVICORN) api.app_api:app --host $(API_HOST) --port $(API_PORT) --reload

.PHONY: ui
ui:
	@if [ -z "$$API_URL" ]; then echo "Set API_URL env (e.g. export API_URL=http://localhost:8000)"; exit 1; fi
	$(PYTHON) ui/app.py

.PHONY: test-api
test-api:
	curl -X POST "http://$(API_HOST):$(API_PORT)/recommend" \
	  -H 'Content-Type: application/json' \
	  -d '{"dataset":"$(DATASET)","user_id":"A3CIUOJXQ5VDQ2","k":10,"fusion":"$(FUSION)", \
	       "w_text":$(W_TEXT),"w_image":$(W_IMAGE),"w_meta":$(W_META), \
	       "use_faiss":true,"faiss_name":"$(FAISS_NAME)","exclude_seen":true}'
