.PHONY: lint format test check clean \
	dashboard dashboard-backend dashboard-frontend \
	backend-lint backend-test

lint:
	ruff check src/
	ruff format --check src/

backend-lint:
	ruff check apps/backend-py/src apps/backend-py/scripts apps/backend-py/tests

format:
	ruff format src/

test:
	pytest tests/ -v --tb=short

backend-test:
	cd apps/backend-py && pytest tests/ -v --tb=short

check: lint test backend-lint backend-test
	@echo "All checks passed."

# Start the FastAPI backend on the first free port from 8000.
dashboard-backend:
	python apps/backend-py/scripts/dev_server.py

# Start the Next.js frontend on the first free port from 3000.
dashboard-frontend:
	cd apps/frontend && npm run dev

# Start both servers together; Ctrl+C stops both.
dashboard:
	@$(MAKE) -j2 dashboard-backend dashboard-frontend

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache build dist *.egg-info
