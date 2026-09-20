from settings import load_timeout


def request_timeout() -> int:
    return load_timeout()
