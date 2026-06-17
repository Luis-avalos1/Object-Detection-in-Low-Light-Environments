# Low-light object detection — reproducible pipeline
PY ?= venv/bin/python

.PHONY: help split benchmark finetune report qualitative test all clean

help:
	@echo "make split       - build deterministic split + YOLO export"
	@echo "make benchmark   - zero-shot detector sweep + enhancement grid"
	@echo "make finetune    - fine-tune YOLOv8n on ExDark (CPU) + before/after"
	@echo "make report      - figures, CSV tables, report.md/html"
	@echo "make qualitative - GT vs prediction panels"
	@echo "make test        - run unit tests"
	@echo "make all         - split -> benchmark -> finetune -> report"

split:
	$(PY) scripts/00_make_split.py

benchmark:
	$(PY) scripts/02_run_benchmark.py

finetune:
	$(PY) scripts/03_finetune.py --base yolov8n.pt --epochs 20 --imgsz 416 --fraction 0.10 --device cpu

report:
	$(PY) scripts/04_make_report.py

qualitative:
	$(PY) scripts/05_qualitative.py

test:
	$(PY) tests/test_metrics.py && $(PY) tests/test_dataset_enhance.py

all: split benchmark finetune report

clean:
	rm -rf runs/ data/exdark_yolo/
