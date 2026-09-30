from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["paid_api_required"] is False
    assert body["demo_mode"] is True


def test_predict_synthetic_sample():
    payload = {
        "order_id": "DEMO-ORDER-001",
        "order_placed_at": "2026-07-01 10:30",
        "customer_id": "DEMO-C005",
        "sku": "DEMO-SKU-04",
        "sales_channel": "web",
        "payment_mode": "cod",
        "discount_pct": 18,
        "qty": 1,
        "order_value_inr": 6500,
        "promised_delivery_days": 7,
        "delivery_pincode": "411001",
        "is_gift": "N",
        "customer_prior_orders": 4,
        "customer_prior_returns": 1,
        "delivery_note": "Synthetic public-demo record",
        "last_service_event_type": "NONE",
        "pickup_scheduled_at": None,
        "source": "demo",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["demo_mode"] is True
    assert 0 <= body["return_probability"] <= 1
    assert body["recommended_action"] in {"SHIP_NORMALLY", "CALL_BEFORE_DISPATCH"}
    assert body["reasons"]
