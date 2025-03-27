from pathlib import Path
from pydantic import BaseModel
from decouple import config as env_config


class Settings(BaseModel):
    """Configuration settings for the application."""

    # Debug/Development options
    DEBUG_MODE: bool = False
    BASE_DIR: Path = Path(__file__).parent.parent


settings = Settings()

print(f"Base directory: {settings.BASE_DIR}")
