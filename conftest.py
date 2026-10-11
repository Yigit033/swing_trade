"""Global pytest safety guard.

Unit/integration tests must never inherit the production DATABASE_URL from the
developer's .env file. Modules call load_dotenv() during import, so seed a
non-PostgreSQL value before test collection begins. Tests that need a specific
database already monkeypatch their own SQLite path.
"""

import os


os.environ["DATABASE_URL"] = "sqlite-test-isolation"
