# Kestrel Home Returns Risk — Public Review Repository

A small FastAPI service for pre-dispatch return-risk scoring, with a browser UI and deterministic human-readable reasons.

> **Privacy note:** This is intentionally a **sanitized public repository**. The assessment pack's customer/operational data, private fitted model, hidden-test predictions, per-order evaluation outputs, policy values, memo and submission form are not committed. The runnable model included here is trained on synthetic demo data only.

## What this repository includes

- `app.py` — `POST /predict`, `/health`, and one browser screen.
- `src/features.py` — decision-time-safe feature engineering.
- `src/explain.py` — deterministic feature-contribution explanations.
- `scripts/build_demo_model.py` — creates the bundled synthetic demo model.
- `scripts/train_authorized.py` — authorized-data training path; requires the original assessment pack and policy values at runtime and writes only to a gitignored private output directory.
- `tests/test_app.py` and `scripts/smoke_test.py` — service checks.
- `PROJECT_NOTES.md` — methodology and design decisions without private metrics/data.

## Run the public demo

Python 3.11+ is recommended. No paid API key is required.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open:

- App: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

## Verify it

```bash
python scripts/smoke_test.py
pytest -q
```

Expected smoke-test line:

```text
SMOKE TEST PASSED
```

## API example

`POST /predict`

```json
{
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
  "pickup_scheduled_at": null,
  "source": "demo"
}
```

The response contains a continuous risk score, an action, and employee-readable reasons. In this public repo, both the model and threshold are synthetic-demo values.

## Rebuild the synthetic demo model

```bash
python scripts/build_demo_model.py
```

This deterministically regenerates `artifacts/demo_model_bundle.joblib` from invented records. It never reads the private assessment data.

## Authorized reproduction path

If you are an authorized reviewer with the original assessment pack, the private training script can reproduce the real fitted model and predictions without placing them in the repository:

```bash
python scripts/train_authorized.py \
  --input-dir /path/to/original-pack \
  --output-dir private_output \
  --return-cost <value-from-policy> \
  --intervention-cost <value-from-policy> \
  --intervention-effectiveness <value-from-policy>
```

`private_output/` is gitignored. Do not commit it.

## Why the confidential artifacts are absent

The supplied assessment policy restricts publication of customer and operational data. To respect that requirement, the following remain outside GitHub and should be delivered through the hiring/assessment portal instead:

- Original labelled/unlabelled/reference data.
- Original policy and email thread.
- Private fitted model bundle.
- `predictions.csv`.
- Per-order validation predictions.
- Exact private evaluation/business figures.
- Client memo and completed submission form.

This split keeps the code reviewable and the app runnable without publishing restricted information.
