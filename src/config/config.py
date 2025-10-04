import os

from pydantic import BaseModel


class Config(BaseModel):
    TELEGRAM_BOT_TOKEN: str
    CLOUD_API_KEY: str
    DATA_FILE: str
    META_FILE: str
    LAST_UPDATE_FILE: str
    RADAR_CACHE_FILE: str
    TRUSTED_SOURCES_FILE: str


def parseConfig() -> Config:
    return Config(
        TELEGRAM_BOT_TOKEN=str(os.getenv('TELEGRAM_BOT_TOKEN')),
        CLOUD_API_KEY=str(os.getenv('CLOUD_API_KEY')),
        DATA_FILE=str(os.getenv('DATA_FILE', "./cache/df_fin.csv")),
        META_FILE=str(os.getenv('META_FILE', "./cache/top_clusters_meta.csv")),
        LAST_UPDATE_FILE=str(os.getenv('LAST_UPDATE_FILE', "./cache/last_update.txt")),
        RADAR_CACHE_FILE=str(os.getenv('RADAR_CACHE_FILE', "./cache/radar_cache.json")),
        TRUSTED_SOURCES_FILE=str(os.getenv('TRUSTED_SOURCES_FILE', "./cache/trusted_sources.json"))
    )

