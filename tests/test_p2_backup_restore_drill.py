import sqlite3
import tempfile
import unittest
from pathlib import Path

from questions_agent_platform.tools.db_backup_restore import (
    collect_row_counts,
    create_backup,
    restore_backup,
    run_backup_restore_drill,
)


class TestP2BackupRestoreDrill(unittest.TestCase):
    def test_sqlite_backup_restore_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "qa.sqlite3"
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
                conn.execute("INSERT INTO sample(value) VALUES (?)", ("alpha",))
                conn.commit()

            database_url = f"sqlite+pysqlite:///{db_path}"
            baseline = collect_row_counts(database_url)
            self.assertEqual(baseline.get("sample"), 1)

            backup = create_backup(database_url=database_url, out_dir=td)
            self.assertTrue(Path(str(backup["backup_path"])).exists())

            with sqlite3.connect(str(db_path)) as conn:
                conn.execute("DELETE FROM sample")
                conn.commit()
            mutated = collect_row_counts(database_url)
            self.assertEqual(mutated.get("sample"), 0)

            restore_backup(
                backup_path=str(backup["backup_path"]),
                target_database_url=database_url,
                clean=True,
            )
            restored = collect_row_counts(database_url)
            self.assertEqual(restored, baseline)

            drill = run_backup_restore_drill(
                source_database_url=database_url,
                out_dir=td,
                fail_on_mismatch=True,
            )
            self.assertEqual(str(drill.get("status")), "pass")
            self.assertTrue(bool(drill.get("restore_checked")))
            self.assertTrue(bool(drill.get("row_counts_match")))


if __name__ == "__main__":
    unittest.main()
