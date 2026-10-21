"""Central settings. Everything else imports from here."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "business.db"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION = "reviews"

# Free LLM via Groq (get a key at https://console.groq.com). Model names change
# over time, so you can override this in your .env file without touching code.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
