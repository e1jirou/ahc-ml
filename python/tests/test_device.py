import pytest
from ahc_ml.device import select_device


def test_select_cpu() -> None:
    device, info = select_device("cpu")
    assert device.type == "cpu"
    assert info.selected == "cpu"
    assert info.requested == "cpu"


def test_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="unsupported device"):
        select_device("tpu")
