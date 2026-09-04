"""End-to-end tests for the scoring service."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from fraud_radar.scoring import DEFAULT_MODEL_PATH

pytestmark = pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.exists(),
    reason="needs a trained model; run scripts/train.py first",
)


@pytest.fixture(scope="module")
def client():
    from api.main import app

    with TestClient(app) as c:
        yield c


def transaction(**overrides) -> dict:
    """A plausible everyday payment, with fields overridable per test."""
    base = {
        "cc_num": 4111111111111111,
        "trans_date_trans_time": "2020-07-15T13:30:00",
        "amt": 42.50,
        "category": "grocery_pos",
        "gender": "F",
        "state": "NC",
        "lat": 36.0788,
        "long": -81.1781,
        "city_pop": 3495,
        "merch_lat": 36.0113,
        "merch_long": -82.0483,
        "dob": "1988-03-09",
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------- health
def test_health_reports_the_loaded_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "v1"
    assert body["model_name"] == "gradient_boosting"


# ---------------------------------------------------------------- score
def test_scores_an_ordinary_transaction(client):
    r = client.post("/score", json=transaction())
    assert r.status_code == 200

    body = r.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert isinstance(body["flagged"], bool)
    assert body["model_version"] == "v1"
    # The definition of done asks for a response in well under 100ms.
    assert body["latency_ms"] < 100


def test_a_suspicious_shape_scores_higher_than_a_normal_one(client):
    """Build a card's habit of small daytime spending, then break it.

    This is the whole point of the history features: the same amount means
    different things on different cards, so the comparison has to be made
    against a card that has an established pattern.
    """
    card = 4222222222222222
    start = datetime(2020, 7, 1, 12, 0, 0, tzinfo=UTC)

    # Twenty small, daytime, local grocery runs.
    for day in range(20):
        ordinary = transaction(
            cc_num=card,
            trans_date_trans_time=(start + timedelta(days=day)).isoformat(),
            amt=30.0 + day % 5,
        )
        assert client.post("/score", json=ordinary).status_code == 200

    baseline = client.post(
        "/score",
        json=transaction(
            cc_num=card,
            trans_date_trans_time=(start + timedelta(days=21)).isoformat(),
            amt=33.0,
        ),
    ).json()

    # Same card: a large online purchase at 11pm, far from home.
    suspicious = client.post(
        "/score",
        json=transaction(
            cc_num=card,
            trans_date_trans_time=(start + timedelta(days=21, hours=11)).isoformat(),
            amt=920.0,
            category="shopping_net",
            merch_lat=40.7128,
            merch_long=-74.0060,
        ),
    ).json()

    assert suspicious["risk_score"] > baseline["risk_score"], (
        f"suspicious {suspicious['risk_score']:.4f} did not exceed "
        f"baseline {baseline['risk_score']:.4f}"
    )


def test_history_accumulates_across_requests(client):
    before = client.get("/health").json()["cards_tracked"]
    client.post("/score", json=transaction(cc_num=4333333333333333))
    after = client.get("/health").json()["cards_tracked"]
    assert after == before + 1


# ----------------------------------------------------------- bad input
def test_missing_field_is_rejected_by_name(client):
    payload = transaction()
    del payload["amt"]

    r = client.post("/score", json=payload)
    assert r.status_code == 422
    assert any(p["field"] == "amt" for p in r.json()["problems"])


def test_out_of_range_values_are_rejected(client):
    r = client.post("/score", json=transaction(lat=500.0))
    assert r.status_code == 422
    assert any(p["field"] == "lat" for p in r.json()["problems"])

    r = client.post("/score", json=transaction(amt=-10.0))
    assert r.status_code == 422
    assert any(p["field"] == "amt" for p in r.json()["problems"])


def test_unknown_category_is_rejected_with_a_useful_message(client):
    r = client.post("/score", json=transaction(category="crypto_rug_pull"))
    assert r.status_code == 422

    problem = next(p for p in r.json()["problems"] if p["field"] == "category")
    assert "grocery_pos" in problem["problem"], "the error should list valid options"


def test_garbage_body_does_not_crash_the_service(client):
    assert client.post("/score", json={"lol": "nope"}).status_code == 422
    assert client.get("/health").status_code == 200
