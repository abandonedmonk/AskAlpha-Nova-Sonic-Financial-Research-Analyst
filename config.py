"""
config.py — Centralised settings loaded from .env
All environment variables are validated here so the app fails fast at startup
rather than mid-request.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── AWS ──────────────────────────────────────────────────────────────────
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str = "us-east-1"

    # ── Nova Sonic ────────────────────────────────────────────────────────────
    nova_sonic_model_id: str = "amazon.nova-sonic-v1:0"

    # ── Bedrock Knowledge Base ────────────────────────────────────────────────
    bedrock_kb_id: str = ""
    bedrock_kb_model_arn: str = (
        "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0"
    )

    # ── Polygon.io ────────────────────────────────────────────────────────────
    polygon_api_key: str

    # ── Finnhub ───────────────────────────────────────────────────────────────
    finnhub_api_key: str = ""
    # ── Tiingo ────────────────────────────────────────────────────────────────────────
    tiingo_api_key: str = ""
    # ── Vault ─────────────────────────────────────────────────────────────────
    vault_path: Path = Path("./vault")

    # ── Ironclad Wasm sandbox ─────────────────────────────────────────────────
    ironclad_runtime_path: Path = Path("./ironclad/ironclad-runtime")

    @property
    def ironclad_available(self) -> bool:
        """True if the ironclad-runtime binary exists on disk."""
        return self.ironclad_runtime_path.exists()

    @property
    def bedrock_kb_configured(self) -> bool:
        """True if a Knowledge Base ID has been provided."""
        return bool(self.bedrock_kb_id)

    # ── App ───────────────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"


# Singleton — import this everywhere
settings = Settings()
