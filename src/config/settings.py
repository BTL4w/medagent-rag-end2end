from dataclasses import dataclass


@dataclass
class Settings:
    app_env: str = "development"
    app_port: int = 8000


settings = Settings()
