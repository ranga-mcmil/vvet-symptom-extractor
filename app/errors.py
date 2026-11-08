"""Application-specific errors."""


class LLMError(Exception):
    """The model failed, timed out, or returned output we cannot use."""