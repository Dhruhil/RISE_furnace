.PHONY: help create-cases create-dataset validate clean test lint

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

create-cases:  ## Generate OpenFOAM parameter study cases
	python -m scripts.create_cases

validate:  ## Validate generated cases before running
	python -m scripts.validate_cases

create-dataset:  ## Build ML training dataset from simulation results
	python -m scripts.create_dataset


clean:  ## Remove generated artifacts
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete

test:  ## Run test suite
	pytest tests/ -v

lint:  ## Run linter
	ruff check src/ scripts/ configs/