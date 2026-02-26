"""Metrics evaluation recipe module."""

from optimus_dl.core.registry import make_registry

from .base import MetricsRecipe
from .config import MetricsConfig

metrics_recipe_registry, register_metrics_recipe, build_metrics_recipe = make_registry(
    "metrics_recipe", MetricsRecipe
)

register_metrics_recipe("base", MetricsConfig)(MetricsRecipe)

__all__ = [
    "MetricsRecipe",
    "MetricsConfig",
    "metrics_recipe_registry",
    "register_metrics_recipe",
    "build_metrics_recipe",
]
