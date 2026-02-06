from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    jwt_secret: str
    file_storage_root: str = "/data/aline_docs"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    redis_url: str = "redis://localhost:6379/0"
    watch_paths: str = "/data/aline_docs"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
