"""Reward function implementations.

Every module in this package is auto-imported at package import time, so a
new reward registers itself simply by declaring
``@register_reward_function`` in a file placed here.
"""

from optimus_dl.core.bootstrap import bootstrap_module

bootstrap_module(__name__)
