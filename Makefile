.PHONY: help install clean check fix typecheck kernel jupyter

help:
	@echo "Available commands:"
	@echo ""
	@echo "Setup"
	@echo "  make install   - Install dependencies"
	@echo "  make clean     - Remove virtual environment"
	@echo ""
	@echo "Quality"

	@echo "  make check     - Run all (fix + typecheck)"

	@echo "  make fix       - Auto-fix and format code"

	@echo "  make typecheck - Run type checking (mypy)"

	@echo ""
	@echo "Jupyter"
	@echo "  make kernel    - Register Jupyter kernel"
	@echo "  make jupyter   - Launch Jupyter Notebook"


# -----------------------------------------
# Setup
# -----------------------------------------

install:
	poetry install

clean:
	poetry env remove python || true

# -----------------------------------------
# Quality
# -----------------------------------------


check: fix typecheck


typecheck:
	poetry run mypy .

fix:
	poetry run ruff check . --fix
	poetry run ruff format .




# -----------------------------------------
# Jupyter
# -----------------------------------------

kernel:
	poetry run python -m ipykernel install --user --name=gut-former --display-name "Python (gut-former)"

jupyter:
	poetry run jupyter notebook

