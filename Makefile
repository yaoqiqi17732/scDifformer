.PHONY: help sync lint

help:
	@echo "Available make targets:"
	@echo "  sync               Install project dependencies using uv"
	@echo "  lint                  Perform static code analysis"
	@echo ""
	@echo "Use 'make <target>' to run a specific command."

sync:
	uv sync

lint:
	uv add pre-commit --group dev && \
	uv run pre-commit run --all-files --verbose
