import importlib
import logging
import os


def bootstrap_module(
    module_name: str,
    path: str | None = None,
    exclude_patterns: list[str] | None = None,
    silent_errors: bool = False,
) -> None:
    """Automatically imports all files and nested submodules of a given module.

    Args:
        module_name: The full module name to import submodules for.
        path: The filesystem path for the module. If None, derived from module_name.
        exclude_patterns: List of patterns to exclude from import.
        silent_errors: If True, log errors instead of raising them.
    """
    if exclude_patterns is None:
        exclude_patterns = ["test_", "tests", "__pycache__", ".pyc"]

    logger = logging.getLogger(__name__)

    try:
        if path is None:
            spec = importlib.util.find_spec(module_name)
            if spec is None or not spec.submodule_search_locations:
                raise ImportError(f"Module '{module_name}' not found or not a package.")
            paths = spec.submodule_search_locations
        else:
            paths = [path]

        for package_path in paths:
            if not os.path.exists(package_path):
                continue

            for entry in os.listdir(package_path):
                # Skip private files and excluded patterns
                if entry.startswith("_") or any(
                    pattern in entry for pattern in exclude_patterns
                ):
                    continue

                full_path = os.path.join(package_path, entry)

                try:
                    if os.path.isdir(full_path):
                        # Check if it's a valid Python package
                        if os.path.exists(os.path.join(full_path, "__init__.py")):
                            submodule_name = f"{module_name}.{entry}"
                            importlib.import_module(submodule_name)
                            logger.debug(f"Imported submodule: {submodule_name}")
                    elif entry.endswith(".py") and entry != "__init__.py":
                        submodule_name = f"{module_name}.{entry[:-3]}"
                        importlib.import_module(submodule_name)
                        logger.debug(f"Imported module: {submodule_name}")

                except ImportError as e:
                    if silent_errors:
                        logger.warning(f"Failed to import {entry}: {e}")
                    else:
                        raise

    except Exception as e:
        if silent_errors:
            logger.error(f"Bootstrap failed for {module_name}: {e}")
        else:
            raise
