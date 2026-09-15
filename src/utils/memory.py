import os
import platform


def get_memory_usage() -> str:
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if platform.system() == "Darwin":
            # On macOS, ru_maxrss is in bytes
            mb = usage.ru_maxrss / (1024 * 1024)
        else:
            # On Linux, ru_maxrss is in KB
            mb = usage.ru_maxrss / 1024
        return f"{mb:.0f} MB"
    except Exception:
        return "unknown"
