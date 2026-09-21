# Presence of a conftest.py at the project root puts this directory on sys.path,
# so `import research` works when running pytest from anywhere in the repo.

import pytest


@pytest.fixture(autouse=True)
def isolate_local_research_history(monkeypatch, tmp_path):
    """Streamlit AppTests must not leave their synthetic runs in user history."""
    monkeypatch.setenv("ALPHAFORGE_HISTORY_PATH", str(tmp_path / "alphaforge-history.db"))
