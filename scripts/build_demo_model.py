from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import CAT_FEATURES, MODEL_FEATURES, NUM_FEATURES, engineer_records

RNG = np.random.default_rng(42)
DEMO_THRESHOLD = 0.25  # synthetic-demo threshold; not a client policy value


def build_refs() -> tuple[dict, dict]:
    cities = ["Pune", "Mumbai", "Nagpur", "Nashik", "Indore"]
    customers = {}
    for i in range(1, 41):
        customers[f"DEMO-C{i:03d}"] = {
            "city": cities[(i - 1) % len(cities)],
            "state": "Demo State",
            "shield_member": "Y" if i % 5 == 0 else "N",
        }

    families = [
        "Air Fryer",
        "Mixer Grinder",
        "Water Purifier",
        "Robot Vacuum",
        "Cooktop",
        "Fan",
        "Heater",
    ]
    products = {}
    for i, family in enumerate(families, start=1):
        products[f"DEMO-SKU-{i:02d}"] = {
            "family": family,
            "list_price_inr": float(2500 + i * 1200),
            "warranty_months": 24 if i in {3, 4} else 12,
            "launch_date": f"2025-{((i - 1) % 9) + 1:02d}-01",
        }
    return customers, products


def make_synthetic_orders(customer_ref: dict, product_ref: dict, n: int = 2400) -> tuple[pd.DataFrame, np.ndarray]:
    customer_ids = np.array(list(customer_ref))
    skus = np.array(list(product_ref))
    channels = np.array(["app", "web", "marketplace", "partner_outlet"])
    payments = np.array(["prepaid_upi", "prepaid_card", "cod", "emi"])

    rows = []
    logits = []
    start = pd.Timestamp("2026-01-01")
    for i in range(n):
        cid = str(RNG.choice(customer_ids))
        sku = str(RNG.choice(skus))
        channel = str(RNG.choice(channels, p=[0.30, 0.34, 0.21, 0.15]))
        payment = str(RNG.choice(payments, p=[0.34, 0.28, 0.25, 0.13]))
        prior_orders = int(RNG.integers(0, 9))
        prior_returns = int(RNG.integers(0, min(3, prior_orders) + 1)) if prior_orders else 0
        discount = float(RNG.integers(0, 31))
        promised = int(RNG.integers(2, 10))
        qty = int(RNG.choice([1, 1, 1, 2]))
        is_gift = "Y" if RNG.random() < 0.08 else "N"
        placed = start + pd.Timedelta(days=int(RNG.integers(0, 240)), minutes=int(RNG.integers(0, 1440)))
        prod = product_ref[sku]
        stored_value = prod["list_price_inr"] * qty * (1 - discount / 100.0)

        rows.append(
            {
                "order_id": f"DEMO-O{i:05d}",
                "order_placed_at": placed.strftime("%Y-%m-%d %H:%M"),
                "customer_id": cid,
                "sku": sku,
                "sales_channel": channel,
                "payment_mode": payment,
                "discount_pct": discount,
                "qty": qty,
                "order_value_inr": stored_value,
                "promised_delivery_days": promised,
                "delivery_pincode": "411001",
                "is_gift": is_gift,
                "customer_prior_orders": prior_orders,
                "customer_prior_returns": prior_returns,
                "delivery_note": "Synthetic public-demo order",
                "last_service_event_type": "NONE",
                "pickup_scheduled_at": None,
                "source": "demo",
            }
        )

        # Synthetic target process. These coefficients are invented solely so the
        # public service has a meaningful, deterministic model to run.
        logit = -3.0
        logit += 0.75 if payment == "cod" else 0.0
        logit += 0.32 if channel == "marketplace" else 0.0
        logit += 0.45 if customer_ref[cid]["shield_member"] == "Y" else 0.0
        logit += 0.90 * (prior_returns / max(prior_orders, 1))
        logit += 0.035 * discount
        logit += 0.10 * max(promised - 4, 0)
        logit += 0.25 if prod["family"] == "Robot Vacuum" else 0.0
        logit += 0.20 if is_gift == "Y" else 0.0
        logits.append(logit)

    probs = 1.0 / (1.0 + np.exp(-np.array(logits)))
    y = RNG.binomial(1, probs)
    return pd.DataFrame(rows), y


def build_model() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("num", StandardScaler(), NUM_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(C=0.1, max_iter=2000, random_state=42)),
        ]
    )


def main() -> None:
    customer_ref, product_ref = build_refs()
    raw, y = make_synthetic_orders(customer_ref, product_ref)
    X = engineer_records(raw, customer_ref, product_ref)
    model = build_model()
    model.fit(X[MODEL_FEATURES], y)

    bundle = {
        "model": model,
        "customer_ref": customer_ref,
        "product_ref": product_ref,
        "call_threshold": DEMO_THRESHOLD,
        "metadata": {
            "demo_only": True,
            "trained_on": "deterministic synthetic data",
            "features": MODEL_FEATURES,
        },
    }
    out = ROOT / "artifacts" / "demo_model_bundle.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out, compress=3)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
