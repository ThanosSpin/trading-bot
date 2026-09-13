import os

from .config_base import *

BOT_ENV = os.getenv("BOT_ENV", "live").strip().lower()

if BOT_ENV not in ("paper", "live", "live2"):
    raise ValueError(
        f"Invalid BOT_ENV={BOT_ENV!r}. "
        "Expected paper, live, or live2."
    )

if BOT_ENV == "paper":
    from .config_paper import *
else:
    from .config_live import *

# Re-assert the process-selected environment
BOT_ENV = os.getenv("BOT_ENV", "live").strip().lower()