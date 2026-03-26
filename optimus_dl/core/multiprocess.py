import logging
import os

import psutil


def force_kill_children(timeout_seconds: int = 15) -> None:
    """
    Recursively terminates all child processes of the current process.
    """
    try:
        current_process = psutil.Process(os.getpid())
        children = current_process.children(recursive=True)
    except psutil.NoSuchProcess:
        return

    # Phase 1: Polite termination
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            continue

    # Phase 2: Force kill if they ignore SIGTERM
    gone, alive = psutil.wait_procs(children, timeout=timeout_seconds)
    for child in alive:
        try:
            logging.warning(f"Force killing unresponsive child process: {child.pid}")
            child.kill()
        except psutil.NoSuchProcess:
            continue
