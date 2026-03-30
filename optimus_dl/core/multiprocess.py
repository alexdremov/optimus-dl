import gc
import logging
import os

import psutil

from .environment import OPTIMUS_EXIT_TIMEOUT

logger = logging.getLogger(__name__)


def finalize_process(timeout_seconds: int = 5, max_retries=10) -> None:
    """
    Finalizes the process by ensuring all child processes are terminated to prevent hanging.
     - First attempts a polite termination of child processes.
     - If any child processes are still alive after the timeout, it forcefully kills them.
     - Schedules a watchdog to force exit the process after a delay as a safety net.

    Args:
        timeout_seconds: Time to wait for child processes to terminate before force killing.
        max_retries: Maximum number of retries to check for alive child processes.
    """
    force_terminate_joblib()
    finish_wandb()
    finish_mlflow()

    children = ["dummy"]
    retries = 0

    while len(children) > 0 and retries < max_retries:
        retries += 1
        gc.collect()
        try:
            current_process = psutil.Process(os.getpid())
            children = current_process.children(recursive=True)
            logger.info(
                f"Found {children} child processes to terminate (retry {retries}/{max_retries})"
            )
        except Exception:
            logger.warning(
                f"Failed to get child processes for PID {os.getpid()}", exc_info=True
            )
            break

        # Phase 1: Polite termination
        for child in children:
            try:
                child.terminate()
            except Exception:
                logger.warning(
                    f"Failed to terminate child process: {child.pid}", exc_info=True
                )
                continue

        # Phase 2: Force kill if they ignore SIGTERM
        _, alive = psutil.wait_procs(children, timeout=timeout_seconds)
        for child in alive:
            try:
                logger.warning(f"Force killing unresponsive child process: {child.pid}")
                child.kill()
            except Exception:
                logger.warning(
                    f"Failed to force kill child process: {child.pid}", exc_info=True
                )
                continue

    _schedule_watchdog()


def _schedule_watchdog():
    """
    Schedules a watchdog to forcefully kill the process after a delay if it is still alive.
    This is a safety net in case some child processes are unresponsive and prevent exit.
    """
    import threading
    import time

    timeout = OPTIMUS_EXIT_TIMEOUT.get()

    if timeout < 0:
        logger.info("Watchdog disabled (OPTIMUS_EXIT_TIMEOUT < 0)")
        return
    elif timeout == 0:
        logger.warning(
            "Immediate watchdog enabled (OPTIMUS_EXIT_TIMEOUT = 0), process will be force killed immediately."
        )
        os._exit(0)  # Force exit without cleanup
        return

    def watchdog():
        time.sleep(timeout)
        logger.warning("Watchdog timeout reached, forcefully exiting process.")
        os._exit(0)  # Force exit without cleanup

    logger.info(
        f"Scheduling watchdog to force exit after {timeout} seconds if not exited."
    )
    threading.Thread(target=watchdog, daemon=True).start()


def force_terminate_joblib():
    """
    Forcefully terminates Joblib workers without allowing them to respawn.
    """
    try:
        import joblib.externals.loky.reusable_executor as loky_reusable_executor
    except ImportError:
        logger.warning(
            "Joblib is not installed, cannot force terminate Joblib workers."
        )
        return

    executor = loky_reusable_executor._executor
    if executor is not None:
        logger.info("Shutting down Joblib reusable executor to prevent worker respawn.")
        logger.warning(
            "Loky workers may not terminate completely. This may cause the process to hang if they do not exit on their own. It is recommended not to use Joblib."
        )
        executor.shutdown(wait=False, kill_workers=True)


def finish_wandb():
    """
    Ensures that all WandB processes are terminated to prevent hanging.
    """
    try:
        import wandb
    except ImportError:
        logger.warning("WandB is not installed, cannot finish WandB run.")
        return

    if wandb.run is not None:
        logger.info("Finishing WandB run to ensure all processes are terminated.")
        wandb.finish()


def finish_mlflow():
    """
    Ensures that all MLflow processes are terminated to prevent hanging.
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow is not installed, cannot finish MLflow run.")
        return

    try:
        mlflow.end_run()
        logger.info("Finished MLflow run to ensure all processes are terminated.")
    except Exception as e:
        logger.warning(f"Failed to finish MLflow run: {e}", exc_info=True)
