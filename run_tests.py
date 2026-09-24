"""Discover and run the project's tests using the minimal pytest shim."""

import importlib.util
import inspect
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # shim/pytest.py
import pytest  # noqa: E402

PROJECT = Path(sys.argv[1]).resolve()
TESTS = PROJECT / "tests"
sys.path.insert(0, str(PROJECT / "Source Code"))
sys.path.insert(0, str(TESTS))

passed, failed = 0, []


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def collect_fixtures(module):
    return {
        name: obj
        for name, obj in vars(module).items()
        if callable(obj) and getattr(obj, "__is_fixture__", False)
    }


def resolve_args(func, fixtures, tmp_root, monkeypatches, cache, skip=()):
    kwargs = {}
    for name in inspect.signature(func).parameters:
        if name == "self" or name in skip:
            continue
        if name == "tmp_path":
            kwargs[name] = Path(tempfile.mkdtemp(dir=tmp_root))
        elif name == "monkeypatch":
            mp = pytest.MonkeyPatch()
            monkeypatches.append(mp)
            kwargs[name] = mp
        elif name in fixtures:
            if name not in cache:
                fixture_kwargs = resolve_args(fixtures[name], fixtures, tmp_root, monkeypatches, cache)
                cache[name] = fixtures[name](**fixture_kwargs)
            kwargs[name] = cache[name]
        else:
            raise RuntimeError(f"Unknown fixture: {name}")
    return kwargs


def run_test(func, instance, fixtures, autouse, module_name, label):
    global passed
    cases = []
    if hasattr(func, "__parametrize__"):
        names, values = func.__parametrize__
        for value in values:
            value = value if isinstance(value, (tuple, list)) else (value,)
            cases.append(dict(zip(names, value)))
    else:
        cases.append({})

    for params in cases:
        monkeypatches, cache = [], {}
        with tempfile.TemporaryDirectory() as tmp_root:
            try:
                for fixture_func in autouse:
                    fixture_kwargs = resolve_args(fixture_func, fixtures, tmp_root, monkeypatches, cache)
                    result = fixture_func(**fixture_kwargs)
                    if inspect.isgenerator(result):
                        next(result)
                kwargs = resolve_args(func, fixtures, tmp_root, monkeypatches, cache, skip=tuple(params))
                kwargs.update(params)
                if instance is not None:
                    func(instance, **kwargs)
                else:
                    func(**kwargs)
                passed += 1
            except Exception:
                failed.append((f"{module_name}::{label}{params or ''}", traceback.format_exc()))
            finally:
                for mp in monkeypatches:
                    mp.undo()


for test_file in sorted(TESTS.glob("test_*.py")):
    module = load(test_file)
    fixtures = collect_fixtures(module)
    autouse = [f for f in fixtures.values() if getattr(f, "__fixture_autouse__", False)]

    for name, obj in vars(module).items():
        if name.startswith("Test") and inspect.isclass(obj):
            instance = obj()
            for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                if method_name.startswith("test_"):
                    run_test(method, instance, fixtures, autouse, test_file.name, f"{name}.{method_name}")
        elif name.startswith("test_") and inspect.isfunction(obj):
            run_test(obj, None, fixtures, autouse, test_file.name, name)

print("=" * 70)
for label, trace in failed:
    print(f"\nFAILED {label}\n{trace}")
print("=" * 70)
print(f"{passed} passed, {len(failed)} failed")
sys.exit(1 if failed else 0)
