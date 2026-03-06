import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

TOP_K = int(os.getenv("TOP_K", "8"))
TOP_SIM_THRESHOLD = float(os.getenv("TOP_SIM_THRESHOLD", "0.15"))

MAX_RAW_MESSAGES = int(os.getenv("MAX_RAW_MESSAGES", "12"))
MAX_TURNS = int(os.getenv("MAX_TURNS", "6"))


#debug
def require_key():
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY in .env")
