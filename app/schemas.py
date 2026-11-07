"""Data models: the fixed list of signs, and the API request/response shapes."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

MAX_MESSAGE_LENGTH = 2000


class SignId(StrEnum):
    """The complete, fixed list of signs. Anything else is invalid."""

    REDUCED_APPETITE = "reduced_appetite"
    WEAKNESS = "weakness"
    SWELLING = "swelling"
    RAPID_BREATHING = "rapid_breathing"
    CLOUDY_EYE = "cloudy_eye"
    FROTH_NOSE_MOUTH = "froth_nose_mouth"
    RED_URINE = "red_urine"
    UNABLE_TO_STAND = "unable_to_stand"
    TICKS = "ticks"


class Value(StrEnum):
    """What the farmer said about a sign."""

    YES = "yes"
    NO = "no"
    NOT_SURE = "not_sure"


class Finding(BaseModel):
    """One sign mentioned in the message."""

    sign_id: SignId
    value: Value
    quote: str = Field(min_length=1)  # the exact words from the message


class ExtractRequest(BaseModel):
    """Body of POST /extract."""

    model_config = ConfigDict(str_strip_whitespace=True)
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)


class ExtractResponse(BaseModel):
    """Body returned by POST /extract."""

    findings: list[Finding]
    escalate: bool
    reason: str