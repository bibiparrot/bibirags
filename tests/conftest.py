import pytest


@pytest.fixture(autouse=True)
def reset_modules():
    """Ensure module-level state is clean between tests."""
    yield
