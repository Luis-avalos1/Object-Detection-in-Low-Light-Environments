# Low-light object detection — reproducible pipeline
PY ?= venv/bin/python

.PHONY: help split benchmark finetune report qualitative test all clean

help:
	@echo "make split       - build deterministic split + YOLO export"
	@echo "make benchmark   - zero-shot detector sweep + enhancement grid"
	@echo "make finetune    - full fine-tune YOLOv8n on ExDark GPU (DEVICE=0) + before/after"
	@echo "make finetune-demo - fast balanced freeze-backbone fine-tune (weak GPU/CPU)"
	@echo "make report      - figures, CSV tables, report.md/html"
	@echo "make qualitative - GT vs prediction panels"
	@echo "make test        - run unit tests"
	@echo "make all         - split -> benchmark -> finetune -> report"

split:
	$(PY) scripts/00_make_split.py

benchmark:
	$(PY) scripts/02_run_benchmark.py

# Full fine-tune on a CUDA GPU (recommended path). --fraction 1.0 uses all
# 5153 train images so no class is starved. Override the GPU with DEVICE=0.
DEVICE ?= 0
finetune:
	$(PY) scripts/03_finetune.py --base yolov8n.pt --epochs 100 --imgsz 640 --fraction 1.0 --device $(DEVICE)

# Fast, class-balanced demo (weak GPU/CPU): freezes the COCO backbone and trains
# on a balanced 120-img/class subset to avoid catastrophic forgetting.
finetune-demo:
	$(PY) scripts/03_finetune.py --base yolov8n.pt --per-class 120 --epochs 24 --imgsz 416 --freeze 10 --device $(DEVICE)

report:
	$(PY) scripts/04_make_report.py

qualitative:
	$(PY) scripts/05_qualitative.py

test:
	$(PY) tests/test_metrics.py && $(PY) tests/test_dataset_enhance.py

all: split benchmark finetune report

clean:
	rm -rf runs/ data/exdark_yolo/
