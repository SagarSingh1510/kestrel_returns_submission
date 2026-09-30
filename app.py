from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.features import engineer_one
from src.explain import reasons_from_linear_model

ROOT = Path(__file__).resolve().parent
BUNDLE = joblib.load(ROOT / "artifacts" / "demo_model_bundle.joblib")
MODEL = BUNDLE["model"]
CUSTOMERS = BUNDLE["customer_ref"]
PRODUCTS = BUNDLE["product_ref"]
DEMO_THRESHOLD = float(BUNDLE["call_threshold"])

app = FastAPI(title="Kestrel Returns Risk - Public Demo", version="1.0")


class OrderRecord(BaseModel):
    order_id: str
    order_placed_at: str = Field(description="Timestamp, e.g. 2026-07-01 10:30")
    customer_id: str
    sku: str
    sales_channel: str
    payment_mode: str
    discount_pct: float
    qty: int
    order_value_inr: float
    promised_delivery_days: int
    delivery_pincode: str | int
    is_gift: str
    customer_prior_orders: int
    customer_prior_returns: int
    delivery_note: Optional[str] = None
    last_service_event_type: Optional[str] = None
    pickup_scheduled_at: Optional[str] = None
    source: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "demo_mode": True,
        "model": "regularized logistic regression trained on synthetic demo data",
        "paid_api_required": False,
    }


@app.post("/predict")
def predict(record: OrderRecord) -> dict:
    raw = record.model_dump()
    try:
        x = engineer_one(raw, CUSTOMERS, PRODUCTS)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    probability = float(MODEL.predict_proba(x)[0, 1])
    reasons = reasons_from_linear_model(MODEL, x, raw)

    if probability >= DEMO_THRESHOLD:
        action = "CALL_BEFORE_DISPATCH"
        action_reason = (
            "Synthetic-demo risk is above the demo decision threshold. "
            "The private submission uses the authorized policy-derived threshold."
        )
    else:
        action = "SHIP_NORMALLY"
        action_reason = "Synthetic-demo risk is below the demo decision threshold."

    return {
        "order_id": record.order_id,
        "return_probability": round(probability, 6),
        "risk_score": round(probability, 6),
        "recommended_action": action,
        "action_reason": action_reason,
        "reasons": reasons,
        "decision_threshold": round(DEMO_THRESHOLD, 6),
        "demo_mode": True,
        "privacy_note": "This public repository uses synthetic reference data and a synthetic fitted model.",
        "model_notes": [
            "Post-outcome service/pickup fields are accepted by the schema but are not model features.",
            "The raw payment-system amount is not used directly by the feature pipeline.",
        ],
    }


PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Kestrel Returns Risk - Public Demo</title>
<style>
body { font-family: Arial, sans-serif; max-width: 980px; margin: 30px auto; padding: 0 18px; color: #222; }
h1 { margin-bottom: 4px; }
.sub { color: #555; margin-top: 0; }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 12px; }
label { display: flex; flex-direction: column; font-size: 13px; gap: 4px; }
input, select { padding: 9px; border: 1px solid #bbb; border-radius: 6px; }
button { margin-top: 16px; padding: 11px 18px; border: 0; border-radius: 7px; cursor: pointer; font-weight: bold; }
#result { white-space: pre-wrap; background: #f5f5f5; padding: 14px; margin-top: 18px; border-radius: 8px; min-height: 70px; }
.note { background: #fff7df; padding: 10px 12px; border-radius: 7px; margin: 14px 0; }
@media(max-width:700px){ .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>Kestrel Home - Return Risk</h1>
<p class="sub">Public, privacy-safe demonstration of the pre-dispatch scoring service.</p>
<div class="note"><strong>Demo mode:</strong> all bundled customer/product references and the fitted model are synthetic. Confidential assessment data, predictions, metrics and policy values are intentionally excluded from this public repository.</div>
<form id="f">
<div class="grid">
<label>Order ID<input name="order_id" value="DEMO-ORDER-001" required></label>
<label>Placed at<input name="order_placed_at" value="2026-07-01 10:30" required></label>
<label>Customer ID<input name="customer_id" value="DEMO-C005" required></label>
<label>SKU<input name="sku" value="DEMO-SKU-04" required></label>
<label>Sales channel<select name="sales_channel"><option>web</option><option>app</option><option>marketplace</option><option>partner_outlet</option></select></label>
<label>Payment mode<select name="payment_mode"><option>prepaid_upi</option><option>prepaid_card</option><option selected>cod</option><option>emi</option></select></label>
<label>Discount %<input name="discount_pct" type="number" value="18" step="0.1"></label>
<label>Quantity<input name="qty" type="number" value="1"></label>
<label>Stored order value (INR)<input name="order_value_inr" type="number" value="6500" step="0.01"></label>
<label>Promised delivery days<input name="promised_delivery_days" type="number" value="7"></label>
<label>Delivery pincode<input name="delivery_pincode" value="411001"></label>
<label>Gift?<select name="is_gift"><option>N</option><option>Y</option></select></label>
<label>Customer prior orders<input name="customer_prior_orders" type="number" value="4"></label>
<label>Customer prior returns<input name="customer_prior_returns" type="number" value="1"></label>
<label>Delivery note<input name="delivery_note" value="Synthetic public-demo record"></label>
</div>
<button type="submit">Score order</button>
</form>
<div id="result">Submit the synthetic sample order to see a prediction.</div>
<script>
const form = document.getElementById('f');
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(form);
  const obj = Object.fromEntries(fd.entries());
  for (const k of ['discount_pct','qty','order_value_inr','promised_delivery_days','customer_prior_orders','customer_prior_returns']) obj[k] = Number(obj[k]);
  const box = document.getElementById('result');
  box.textContent = 'Scoring...';
  try {
    const r = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(obj)});
    const out = await r.json();
    if (!r.ok) throw new Error(JSON.stringify(out));
    box.textContent = JSON.stringify(out, null, 2);
  } catch(err) { box.textContent = 'Error: ' + err.message; }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PAGE
