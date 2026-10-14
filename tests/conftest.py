import io
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app  # noqa: E402
from config import Config  # noqa: E402


@pytest.fixture(scope="session")
def app():
    class TestConfig(Config):
        TESTING = True
        MAX_CONTENT_LENGTH = 5 * 1024 * 1024

    application = create_app(TestConfig)
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture(scope="session")
def model_available(app):
    return os.path.exists(app.config["MODEL_PATH"])


def _png_bytes(size=(128, 128), color=(120, 120, 120)):
    image = Image.new("RGB", size, color)
    for x in range(size[0]):
        for y in range(size[1]):
            image.putpixel((x, y), ((x * 2) % 256, (y * 2) % 256, (x + y) % 256))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    buffer.seek(0)
    return buffer


@pytest.fixture()
def sample_image():
    return _png_bytes()
