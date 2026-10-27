#!/usr/bin/env python3
"""Run the MARCH smoke tests without pytest.

The DiffRwkv repository root contains an ``__init__.py``, which makes pytest treat the whole
tree as one package and fail while importing the repo's unrelated top-level ``models`` package.
This runner imports the test module by path instead.

    python code/march/run_tests.py
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # so `import march` works


def main() -> int:
    path = HERE / "tests" / "test_march.py"
    spec = importlib.util.spec_from_file_location("test_march", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    tests = [n for n in dir(module) if n.startswith("test_")]
    failed = 0
    for name in sorted(tests):
        try:
            getattr(module, name)()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001 - report every failure
            failed += 1
            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
