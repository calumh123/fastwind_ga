import os
from contextlib import contextmanager


@contextmanager
def go_to(destination):
    "Context manager to change dir and return"
    try:
        cwd = os.getcwd()
        os.chdir(destination)
        yield
    finally:
        os.chdir(cwd)
