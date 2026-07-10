"""Unit tests for the CNNClassifier LightningModule (forward / loss / shapes)."""
import pytest
import torch

from src.model import CNNClassifier

pytestmark = [pytest.mark.training]



def _tiny_model(in_channels: int = 3, num_classes: int = 4) -> CNNClassifier:
    """A small, fast CNNClassifier for unit testing."""
    return CNNClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        num_blocks=2,
        base_channels=8,
        hidden_dim=16,
    )


def test_forward_output_shape():
    model = _tiny_model().eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()


def test_num_classes_controls_output_dim():
    model = _tiny_model(num_classes=7).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 32, 32))
    assert out.shape == (1, 7)


def test_in_channels_adapts():
    model = _tiny_model(in_channels=5).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 5, 32, 32))
    assert out.shape == (1, 4)


def test_default_criterion_produces_scalar_loss():
    model = _tiny_model()
    logits = model(torch.randn(4, 3, 32, 32))
    loss = model.criterion(logits, torch.randint(0, 4, (4,)))
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_class_names_default_to_num_classes():
    model = _tiny_model(num_classes=4)
    assert len(model.class_names) == 4


def test_backward_produces_gradients():
    model = _tiny_model()
    logits = model(torch.randn(2, 3, 32, 32))
    loss = model.criterion(logits, torch.randint(0, 4, (2,)))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)
