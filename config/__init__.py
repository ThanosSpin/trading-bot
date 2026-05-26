import os

env = os.getenv("BOT_ENV", "live").lower()

if env == "paper":
    from .config_paper import *
else:
    from .config_live import *