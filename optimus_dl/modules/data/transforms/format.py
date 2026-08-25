import logging
from dataclasses import dataclass

import torchdata.nodes
from omegaconf import MISSING
from torchdata.nodes.base_node import BaseNode

from optimus_dl.core.registry import RegistryConfigStrict
from optimus_dl.modules.data.transforms import (
    BaseTransform,
    register_transform,
)

logger = logging.getLogger(__name__)


@dataclass
class FormatTransformConfig(RegistryConfigStrict):
    """Configuration for formatting text.

    Attributes:
        template: A string template (e.g., "Question: {question}\nAnswer:")
        output_field: The field to store the formatted text.
    """

    template: str = MISSING
    output_field: str = "text"


@register_transform("format", FormatTransformConfig)
class FormatTransform(BaseTransform):
    """Transform that formats multiple fields into a single string using a template."""

    def __init__(self, cfg: FormatTransformConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.template = cfg.template
        self.output_field = cfg.output_field

    def _map(self, sample):
        """Format the sample."""
        try:
            formatted_text = self.template.format(**sample)
            return {
                **sample,
                self.output_field: formatted_text,
            }
        except KeyError as e:
            logger.error(f"Missing key for format template: {e}. Sample: {sample}")
            raise

    def build(self, source: BaseNode) -> BaseNode:
        return torchdata.nodes.Mapper(
            source=source,
            map_fn=self._map,
        )
