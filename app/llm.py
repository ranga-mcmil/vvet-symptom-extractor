"""LLM client"""

from collections.abc import Callable
from typing import Any

# An "LLM function" takes the farmer's message and returns the model's raw dict.
LLMFunc = Callable[[str], dict[str, Any]]