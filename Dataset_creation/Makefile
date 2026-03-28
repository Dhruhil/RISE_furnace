.PHONY: help create-cases create-dataset create-all-regions-dataset merge-all-regions-datasets validate clean test lint

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

create-cases:  ## Step 1: Generate OpenFOAM parameter study cases
	python -m scripts.create_cases

validate:  ## Step 2: Validate generated cases before running
	python -m scripts.validate_cases

create-dataset:  ## Step 4: Build steel-only ML dataset (dataset_cylinder_features.h5)
	python -m scripts.create_dataset

create-all-regions-dataset:  ## Step 5: Build all-regions dataset (dataset_all_regions.h5)
	python -m scripts.create_all_regions_dataset

merge-all-regions-datasets:  ## Step 6: Merge multiple all-regions .h5 files
	python -m scripts.merge_all_regions_datasets $(INPUTS) --output $(OUTPUT)


clean:  ## Remove generated artifacts
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete

test:  ## Run test suite
	pytest tests/ -v

lint:  ## Run linter
	ruff check src/ scripts/ configs/