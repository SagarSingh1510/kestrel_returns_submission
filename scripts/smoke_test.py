from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

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

health = client.get("/health")
assert health.status_code == 200, health.text
prediction = client.post("/predict", json=payload)
assert prediction.status_code == 200, prediction.text
assert prediction.json()["demo_mode"] is True
print("SMOKE TEST PASSED")
print(prediction.json())
