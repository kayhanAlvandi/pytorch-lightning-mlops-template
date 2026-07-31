"""Tests for DBLogger.log_image_metadata.

Requires a running PostgreSQL instance with the schema from
database/init/01_prediction.sql applied.

Set the DB_TEST_URI environment variable to point to your test database, e.g.:
    DB_TEST_URI=postgresql://postgres:postgres@localhost:5432/image_classifier_test
"""
import os
import pytest
import psycopg

from database.dblogger import DBLogger

DB_USER = "admin"
DB_PASS = "admin123"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "prediction_test"

DB_TEST_URI = os.getenv(
    "DB_TEST_URI",
    f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
)
# URI to the default 'postgres' DB (used to create/drop the test DB)
DB_ADMIN_URI = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/postgres"

SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "database", "init", "01_prediction.sql"
)


def setup_module():
    """Apply schema in a fresh public schema."""
    conn = psycopg.connect(DB_TEST_URI, autocommit=True)
    conn.execute("DROP SCHEMA public CASCADE")
    conn.execute("CREATE SCHEMA public")
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        conn.execute(f.read())
    conn.close()


def teardown_module():
    """Drop the test database after all tests finish."""
    conn = psycopg.connect(DB_TEST_URI, autocommit=True)
    conn.execute("DROP SCHEMA public CASCADE")
    conn.execute("CREATE SCHEMA public")
    conn.close()


@pytest.fixture
def db_logger():
    logger = DBLogger(db_uri=DB_TEST_URI)
    logger.connect()
    yield logger
    # Truncate between tests for isolation, then close
    logger.connection.execute("TRUNCATE image_metadata CASCADE")
    logger.connection.commit()
    logger.connection.close()


def _sample_image(overrides: dict | None = None) -> dict:
    """Return a sample image_metadata dict."""
    data = {
        "plate": "plate_001",
        "well": "A01",
        "field": 1,
        "channel": 1,
        "root_path": "/data/images",
        "file_name": "plate_001_A01_f1_ch1.tif",
        "shape_x": 2048,
        "shape_y": 2048,
    }
    if overrides:
        data.update(overrides)
    return data


class TestLogImageMetadata:
    """Tests for DBLogger.log_image_metadata."""

    def test_insert_returns_id(self, db_logger):
        """First insert should return a positive integer id."""
        image = _sample_image()
        image_id = db_logger.log_image_metadata(image)
        assert image_id is not None
        assert isinstance(image_id, int)
        assert image_id > 0

    def test_duplicate_returns_same_id(self, db_logger):
        """Inserting the same file_name twice should return the same id (ON CONFLICT)."""
        image = _sample_image()
        id_first = db_logger.log_image_metadata(image)
        id_second = db_logger.log_image_metadata(image)
        assert id_first == id_second

    def test_duplicate_updates_root_path(self, db_logger):
        """ON CONFLICT should update root_path to the new value."""
        image = _sample_image()
        db_logger.log_image_metadata(image)

        image["root_path"] = "/new/path"
        db_logger.log_image_metadata(image)

        with db_logger.connection.cursor() as cur:
            cur.execute(
                "SELECT root_path FROM image_metadata WHERE file_name = %s",
                (image["file_name"],),
            )
            row = cur.fetchone()
        assert row[0] == "/new/path"

    def test_different_files_get_different_ids(self, db_logger):
        """Two different file_names should get distinct ids."""
        img1 = _sample_image({"file_name": "img_ch1.tif", "channel": 1})
        img2 = _sample_image({"file_name": "img_ch2.tif", "channel": 2})
        id1 = db_logger.log_image_metadata(img1)
        id2 = db_logger.log_image_metadata(img2)
        assert id1 != id2

    def test_data_persisted_correctly(self, db_logger):
        """All columns should be stored correctly."""
        image = _sample_image()
        image_id = db_logger.log_image_metadata(image)

        with db_logger.connection.cursor() as cur:
            cur.execute(
                "SELECT plate, well, field, channel, root_path, file_name, shape_x, shape_y "
                "FROM image_metadata WHERE id = %s",
                (image_id,),
            )
            row = cur.fetchone()

        assert row[0] == image["plate"]
        assert row[1] == image["well"]
        assert row[2] == image["field"]
        assert row[3] == image["channel"]
        assert row[4] == image["root_path"]
        assert row[5] == image["file_name"]
        assert row[6] == image["shape_x"]
        assert row[7] == image["shape_y"]
