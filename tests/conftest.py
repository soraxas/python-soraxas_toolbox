"""Pytest configuration and shared fixtures."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image_path(temp_dir):
    """Create a sample image file for testing."""
    try:
        import numpy as np
        from PIL import Image

        img_path = temp_dir / "test_image.png"
        # Create a simple 100x100 RGB image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(img_path)
        return str(img_path)
    except ImportError:
        pytest.skip("PIL/Pillow not available")


@pytest.fixture
def sample_numpy_image():
    """Create a sample numpy image array."""
    try:
        import numpy as np

        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    except ImportError:
        pytest.skip("numpy not available")


@pytest.fixture
def sample_pil_image():
    """Create a sample PIL image."""
    try:
        import numpy as np
        from PIL import Image

        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return Image.fromarray(img_array)
    except ImportError:
        pytest.skip("PIL/Pillow not available")


@pytest.fixture
def sample_grayscale_numpy_image():
    """Create a sample grayscale numpy image array."""
    try:
        import numpy as np

        return np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    except ImportError:
        pytest.skip("numpy not available")


@pytest.fixture
def mock_terminal_env(monkeypatch):
    """Mock terminal environment variables."""
    monkeypatch.setenv("TERMINAL_WIDTH", "80")
    monkeypatch.setenv("TERMINAL_HEIGHT", "24")
    yield
    monkeypatch.delenv("TERMINAL_WIDTH", raising=False)
    monkeypatch.delenv("TERMINAL_HEIGHT", raising=False)
