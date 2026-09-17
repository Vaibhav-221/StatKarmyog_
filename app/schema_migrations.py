"""Small SQLite schema upgrades for the prototype database.

The project does not use Alembic; these migrations keep existing local demo
databases compatible with newer SQLAlchemy models without deleting seed data.
"""

from __future__ import annotations

from sqlalchemy.engine import Engine


def ensure_lightweight_schema_upgrades(engine: Engine) -> None:
    """Add nullable columns needed by newer code when an older SQLite DB exists."""
    if not str(engine.url).startswith("sqlite"):
        return

    with engine.begin() as conn:
        quiz_rows = conn.exec_driver_sql("PRAGMA table_info(quiz_attempts)").fetchall()
        quiz_columns = {row[1] for row in quiz_rows}
        if quiz_rows and "artifact_id" not in quiz_columns:
            conn.exec_driver_sql("ALTER TABLE quiz_attempts ADD COLUMN artifact_id VARCHAR")
        if quiz_rows and "target_competency" not in quiz_columns:
            conn.exec_driver_sql("ALTER TABLE quiz_attempts ADD COLUMN target_competency VARCHAR")

        officer_rows = conn.exec_driver_sql("PRAGMA table_info(officers)").fetchall()
        officer_columns = {row[1] for row in officer_rows}
        if officer_rows and "profile_photo_url" not in officer_columns:
            conn.exec_driver_sql("ALTER TABLE officers ADD COLUMN profile_photo_url VARCHAR")
