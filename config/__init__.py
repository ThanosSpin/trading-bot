import os

BOT_ENV = os.getenv("BOT_ENV", "live").strip().lower()

if BOT_ENV not in ("paper", "live", "live2"):
    raise ValueError(
        f"Invalid BOT_ENV='{BOT_ENV}'. "
        "Expected: paper, live, or live2."
    )

if BOT_ENV == "paper":
    from .config_paper import *
elif BOT_ENV in ("live", "live2"):
    from .config_live import *