from pydantic_settings import BaseSettings
from functools import lru_cache

class APIFootballSettings(BaseSettings):
    api_football_key: str = "226a6e582d3a1ab16ccac616c63f304c"
    api_football_base_url: str = "https://v3.football.api-sports.io/"
    api_football_timeout: int = 30
    cache_ttl: int = 3600  # 1 heure en secondes
    max_retries: int = 3
    rate_limit_per_minute: int = 30  # Ajuster selon votre plan

    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_api_football_settings():
    return APIFootballSettings()