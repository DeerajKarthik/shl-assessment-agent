import pytest
from app.settings import Settings
from app.service import RecommenderService
import hypothesis

hypothesis.settings.register_profile("default", deadline=None)
hypothesis.settings.load_profile("default")

@pytest.fixture
def service():
    settings = Settings()
    return RecommenderService(settings)
