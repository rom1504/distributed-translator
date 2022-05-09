import pytest
from distributed_translator import hello_world


@pytest.mark.parametrize("message", ["hello", "world"])
def test_hello_world(message):
    hello_world(message)
