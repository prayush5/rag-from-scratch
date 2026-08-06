from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    JINA_API_KEY: str
    EMBEDDING_MODEL: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

settings = Settings()