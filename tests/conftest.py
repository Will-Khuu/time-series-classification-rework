import pytest

from arwsn.loading import DATA_DIR


@pytest.fixture(scope="session")
def data_dir():
    if not DATA_DIR.exists():
        pytest.skip("AReM data missing; run python data/download.py")
    return DATA_DIR
