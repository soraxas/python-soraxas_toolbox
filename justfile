@default: help

# Show all available commands with descriptions
@help:
    just --list

# Sync all workspace packages with uv
# Run this after pulling changes or adding new dependencies
[group('maintenance')]
uvs:
    uv sync --all-packages

# Run unit tests with coverage
[group('test')]
testcov:
    uv run coverage run -m pytest --junitxml=junit.xml -o junit_family=legacy
    # uv run coverage html
    # uv run coverage lcov
    uv run coverage combine
    uv run coverage report

# Run unit tests
[group('test')]
test:
    uv run pytest
