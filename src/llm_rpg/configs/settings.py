from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(".env"))


class Settings(BaseSettings):
    models_path: str
    assets_path: str
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # This allows MODELS_PATH to match models_path
    )


# This should now work if MODELS_PATH is defined in your .env file
settings = Settings()
