PY ?= python3

.PHONY: all figures model time scale clean

all: figures

figures: model time scale

model:
	$(PY) src/model.py > /dev/null

time:
	$(PY) src/figure_time.py > /dev/null

scale:
	$(PY) src/figure_scale.py > /dev/null

# figures/ and results/ are committed deliverables; clean only removes caches.
clean:
	rm -rf src/__pycache__ .pytest_cache .mypy_cache .ruff_cache
