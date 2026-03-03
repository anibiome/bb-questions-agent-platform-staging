from __future__ import annotations

import re
import unittest
from pathlib import Path


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_facade_modules_remain_thin_and_explicit(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expectations = {
            root / "pipeline" / "api.py": "from . import api_monolith as _legacy",
            root / "pipeline" / "service.py": "from . import service_monolith as _legacy",
            root / "prod" / "service_pg.py": "from . import service_pg_monolith as _legacy",
        }
        for path, marker in expectations.items():
            if not path.exists():
                # Support intermediate refactors where a facade has not landed yet.
                continue
            text = path.read_text(encoding="utf-8")
            self.assertIn(marker, text, f"Missing facade marker in {path}")
            self.assertIn("__all__", text, f"Facade should expose explicit __all__ in {path}")

    def test_only_facades_import_monolith_modules(self) -> None:
        root = Path(__file__).resolve().parents[1]
        py_files = [
            p
            for p in root.rglob("*.py")
            if ".git" not in p.parts and "__pycache__" not in p.parts
        ]
        banned_import = re.compile(
            r"(from\s+questions_agent_platform\.(pipeline\.(api_monolith|service_monolith)|prod\.service_pg_monolith)\s+import|"
            r"import\s+questions_agent_platform\.(pipeline\.(api_monolith|service_monolith)|prod\.service_pg_monolith)|"
            r"from\s+\.\s+import\s+(api_monolith|service_monolith|service_pg_monolith))"
        )

        allowlist = {
            root / "pipeline" / "api.py",
            root / "pipeline" / "service.py",
            root / "prod" / "service_pg.py",
            root / "pipeline" / "api_monolith.py",
            root / "pipeline" / "service_monolith.py",
            root / "prod" / "service_pg_monolith.py",
            root / "tests" / "test_architecture_boundaries.py",
        }

        offenders: list[str] = []
        for path in py_files:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # Ignore accidental binary/test fixture blobs with .py extension.
                continue
            if banned_import.search(text) and path not in allowlist:
                offenders.append(str(path.relative_to(root)))

        self.assertEqual([], offenders, f"Monolith import boundary violation(s): {offenders}")


if __name__ == "__main__":
    unittest.main()
