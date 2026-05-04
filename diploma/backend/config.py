from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class Config:
    # Фиксируем CPU для твоего случая
    device: str = "cpu"
    model_path: Path = Path(os.getenv("MODEL_PATH", "models/ProECGNetbeststtc.pth"))
    qdrant_url: str = os.getenv("QDRANT_URL", "https://demo.qdrant.cloud")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "demo")
    qdrant_collection: str = "kr_production_cloud"
    gigachat_credentials: str = os.getenv("GIGACHAT_CREDENTIALS", "MTUyNzYzZDctZWZjOS00NjdlLTkxN2UtODhhMWU3ZDVkYzgzOmRiMGZmMTllLWVjNjYtNGNkOS05ZmQ4LTU1ZTE2ZDVhODAyZQ==")
    verify_ssl: bool = False

config = Config()
print(f"✅ Config loaded: device={config.device}")
