"""Very small subset of pytest, used only to execute this project's tests in an
offline container where the real pytest wheel cannot be installed."""

import contextlib
import inspect
import os
import re
import tempfile
from pathlib import Path


class _Approx:
    def __init__(self, expected, rel=None, abs=None):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def _tol(self):
        if self.abs is not None:
            return self.abs
        if self.rel is not None:
            return abs(self.expected) * self.rel
        return max(1e-6, abs(self.expected) * 1e-6)

    def __eq__(self, other):
        return abs(float(other) - float(self.expected)) <= self._tol()

    def __repr__(self):
        return f"approx({self.expected})"


def approx(expected, rel=None, abs=None):
    return _Approx(expected, rel, abs)


@contextlib.contextmanager
def raises(exc_type, match=None):
    try:
        yield
    except exc_type as error:
        if match and not re.search(match, str(error)):
            raise AssertionError(f"pattern {match!r} not found in {str(error)!r}") from None
    else:
        raise AssertionError(f"DID NOT RAISE {exc_type}")


def fixture(func=None, **kwargs):
    def wrap(f):
        f.__is_fixture__ = True
        f.__fixture_autouse__ = kwargs.get("autouse", False)
        return f

    return wrap(func) if func else wrap


class _Mark:
    @staticmethod
    def parametrize(argnames, argvalues):
        names = [n.strip() for n in argnames.split(",")]

        def decorator(func):
            func.__parametrize__ = (names, argvalues)
            return func

        return decorator

    def __getattr__(self, name):
        def passthrough(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

        return passthrough


mark = _Mark()


class MonkeyPatch:
    def __init__(self):
        self._saved = []

    def setenv(self, name, value):
        self._saved.append((name, os.environ.get(name)))
        os.environ[name] = str(value)

    def delenv(self, name, raising=True):
        self._saved.append((name, os.environ.get(name)))
        if name in os.environ:
            del os.environ[name]
        elif raising:
            raise KeyError(name)

    def setattr(self, target, name, value):
        self._saved.append((None, (target, name, getattr(target, name, None))))
        setattr(target, name, value)

    def undo(self):
        for name, old in reversed(self._saved):
            if name is None:
                target, attr, value = old
                setattr(target, attr, value)
            elif old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old
        self._saved = []
