import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


class Settings:
    """
    Centralized configuration management for Legal Assist RAG.
    All environment variables and constants should be declared here.
    """

    # === Project Paths === #
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"

    # === Vector Store (Milvus) === #
    MILVUS_DB_PATH: str = os.getenv("MILVUS_DB_PATH")  # Local file path
    MILVUS_HOST: str = os.getenv("MILVUS_HOST")
    MILVUS_PORT: str = os.getenv("MILVUS_PORT")
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "ipc_sections")

    # === LLM / Embedding Model Configs === #
    HF_API_KEY: str = os.getenv("HF_API_KEY")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

    # === Chunking Parameters === #
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 100))

    # === Logging / Debug === #
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"


# Instantiate global settings object
settings = Settings()
