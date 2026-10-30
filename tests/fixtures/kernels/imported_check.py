"""An imported comparison must still earn numerical verification."""
from torch import allclose as close
from _base import add_kernel, launch, make_inputs

if __name__ == '__main__':
    x, y = make_inputs()
    out = launch(add_kernel, x, y)
    assert close(out, x + y, atol=1e-5, rtol=1e-5), 'verification failed'
