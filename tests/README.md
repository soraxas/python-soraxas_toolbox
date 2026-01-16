# Test Suite for soraxas_toolbox

## Overview

This directory contains comprehensive unit tests for the `soraxas_toolbox` package. The test suite is built using [pytest](https://docs.pytest.org/).

## Setup

### Install Dependencies

To run the tests, you need to install the package and its test dependencies:

```bash
# Install the package in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"

# For full test coverage, install optional dependencies
pip install -e ".[dev,typecheck-lite,typecheck-all]"
```

### Running Tests

Run all tests:

```bash
pytest
```

Run tests for a specific module:

```bash
pytest tests/test_image.py
```

Run tests with verbose output:

```bash
pytest -v
```

Run tests with coverage:

```bash
pytest --cov=soraxas_toolbox --cov-report=html
```

Run only fast tests (skip slow tests):

```bash
pytest -m "not slow"
```

Run tests for specific markers:

```bash
pytest -m "requires_torch"
pytest -m "requires_matplotlib"
```

## Test Structure

### `test_image.py`

Comprehensive unit tests for `soraxas_toolbox.image` module covering:

- **Image I/O**: `read_as_array`, `plt_fig_to_nparray`
- **Image Display**: `display`, `TerminalImageViewer`, `DisplayableImage`
- **Image Processing**: `resize`, `normalise`, `ensure_uint8_image`, `make_displayable_image`
- **Type Conversions**: `ensure_is_numpy`, `ensure_is_pillow`
- **Array Fixers**: `NumpyArrayAutoFixer`, `TorchArrayAutoFixer`
- **Image Concatenation**: `concat_images`
- **Utility Functions**: `get_new_shape_maintain_ratio`, `cumulative_sum_starts_at`
- **Specialized Functions**: `view_high_dimensional_embeddings`, `dot_to_image`

### Test Coverage

The test suite aims for comprehensive branch coverage, testing:

- All code paths and conditional branches
- Edge cases and error conditions
- Different input types (numpy arrays, PIL images, torch tensors)
- Optional dependencies (torch, matplotlib, PIL, etc.)

## Test Markers

The following pytest markers are used:

- `@pytest.mark.slow`: Tests that take longer to run
- `@pytest.mark.requires_torch`: Tests requiring PyTorch
- `@pytest.mark.requires_cv2`: Tests requiring OpenCV
- `@pytest.mark.requires_matplotlib`: Tests requiring matplotlib
- `@pytest.mark.requires_pil`: Tests requiring PIL/Pillow
- `@pytest.mark.requires_term_image`: Tests requiring term_image
- `@pytest.mark.requires_timg`: Tests requiring timg binary

## Continuous Integration

Tests are designed to work in CI environments where optional dependencies may not be available. Tests that require optional dependencies will be skipped automatically if the dependency is not installed.

## Contributing

When adding new features:

1. Add corresponding unit tests
2. Ensure all branches are covered
3. Test with and without optional dependencies
4. Use appropriate markers for test categorization
