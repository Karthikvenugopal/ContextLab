import os


def load_timeout() -> int:
    return int(os.environ.get("APP_TIMEOUT", "30"))
