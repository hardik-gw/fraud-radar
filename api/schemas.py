"""What a valid transaction looks like coming in, and what goes back out.

Pydantic turns these classes into request validation. A payload missing a field,
carrying a latitude of 500, or a negative amount is rejected with a 422 and a
message naming the offending field — before any of it reaches the model.

That matters more than it sounds. A model handed nonsense does not raise an
error; it returns a confident score computed from nonsense.
"""

from datetime import UTC, date, datetime

from pydantic import BaseModel, Field, field_validator

# The 14 categories the model was trained on. Anything else is either a typo or
# a genuinely new merchant type — both worth rejecting loudly rather than
# silently one-hot encoding to all zeros.
KNOWN_CATEGORIES = {
    "entertainment", "food_dining", "gas_transport", "grocery_net",
    "grocery_pos", "health_fitness", "home", "kids_pets", "misc_net",
    "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel",
}


class Transaction(BaseModel):
    """One card payment, as the outside world sends it."""

    cc_num: int = Field(..., description="Card number, used only as a history key")
    trans_date_trans_time: datetime
    amt: float = Field(..., gt=0, le=100_000, description="Amount in dollars")
    category: str
    gender: str = Field(..., pattern="^[MF]$")
    state: str = Field(..., min_length=2, max_length=2)
    lat: float = Field(..., ge=-90, le=90, description="Cardholder home latitude")
    long: float = Field(..., ge=-180, le=180)
    city_pop: int = Field(..., ge=0)
    merch_lat: float = Field(..., ge=-90, le=90)
    merch_long: float = Field(..., ge=-180, le=180)
    dob: date

    @field_validator("category")
    @classmethod
    def known_category(cls, v: str) -> str:
        if v not in KNOWN_CATEGORIES:
            raise ValueError(
                f"unknown category {v!r}; expected one of "
                f"{', '.join(sorted(KNOWN_CATEGORIES))}"
            )
        return v

    @field_validator("dob")
    @classmethod
    def plausible_dob(cls, v: date) -> date:
        # tz-aware "today" so the check behaves the same wherever it runs
        if v.year < 1900 or v > datetime.now(UTC).date():
            raise ValueError("date of birth is not plausible")
        return v


class ScoreResponse(BaseModel):
    """What the service sends back."""

    risk_score: float = Field(..., description="0 to 1; higher is more suspicious")
    flagged: bool = Field(..., description="True when risk_score >= threshold")
    threshold: float
    model_version: str
    model_name: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_name: str
    cards_tracked: int
