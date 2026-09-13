import os

BOT_ENV = os.getenv("BOT_ENV", "live").strip().lower()

if BOT_ENV not in ("paper", "live", "live2"):
    raise ValueError(
        f"Invalid BOT_ENV={BOT_ENV!r}. "
        "Expected paper, live, or live2."
    )

# ------------------------------------------------------------
# Shared strategy/config values
# ------------------------------------------------------------
from .config_base import *

# ------------------------------------------------------------
# Environment-specific overrides
# ------------------------------------------------------------
if BOT_ENV == "paper":
    from .config_paper import *
else:
    from .config_live import *

# Re-assert BOT_ENV because wildcard imports may contain
# another BOT_ENV definition.
BOT_ENV = os.getenv("BOT_ENV", "live").strip().lower()
