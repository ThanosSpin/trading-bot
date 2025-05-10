import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

SYMBOL = 'NVDA'
CASH = 1500
THRESHOLD = 0.05