PYTHON = python3
VENV = venv
SYL = syl.txt
PORT = 5005

ifeq ($(OS), Windows_NT)
	PY = $(VENV)/Scripts/python.exe
else
	PY = $(VENV)/bin/python
endif

.DEFAULT_GOAL := help

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "  setup    install dependencies"
	@echo "  desktop  run desktop app"
	@echo "  web      run web app"
	@echo "  browse   open dataset browser"
	@echo "  convert  generate all stroke images"
	@echo "  clean    remove generated images"

setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -r requirements.txt

desktop:
	$(PY) hw_collector.py --file $(SYL)

web:
	$(PY) server.py --file $(SYL) --port $(PORT)

browse:
	$(PY) dataset_browser.py --dataset dataset --textfile $(SYL)

convert:
	$(PY) convert2image.py --dataset dataset --output single --color_mode single
	$(PY) convert2image.py --dataset dataset --output stroke --color_mode stroke
	$(PY) convert2image.py --dataset dataset --output time   --color_mode time

clean:
	rm -rf single stroke time

.PHONY: help setup desktop web browse images clean