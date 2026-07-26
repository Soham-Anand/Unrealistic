import os
import platform


def get_memory_usage() -> str:
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        mb = usage.ru_maxrss / 1024
        return f"{mb:.0f} MB"
    except Exception:
        return "unknown"
