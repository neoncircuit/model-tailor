.PHONY: lint format test check clean

lint:
	ruff check src/
	ruff format --check src/

format:
	ruff format src/

test:
	pytest tests/ -v --tb=short

check: lint test
	@echo "All checks passed."

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache build dist *.egg-info
