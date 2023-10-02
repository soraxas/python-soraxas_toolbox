import pytest

torch = pytest.importorskip("torch")

from soraxas_toolbox.image import TorchArrayAutoFixer


def test_detect_tensor():
    assert not TorchArrayAutoFixer.infer_is_batch(torch.rand(3, 3))
    # gray scale
    assert TorchArrayAutoFixer.infer_is_batch(torch.rand(10, 10, 10))
    # infer as RGB
    assert not TorchArrayAutoFixer.infer_is_batch(torch.rand(3, 10, 10))
    assert TorchArrayAutoFixer.infer_is_batch(torch.rand(10, 3, 5, 3))
    assert TorchArrayAutoFixer.infer_is_batch(torch.rand(2, 10, 3, 5, 3))
