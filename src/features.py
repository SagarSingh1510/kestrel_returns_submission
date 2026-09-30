from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

CAT_FEATURES = [
    "sales_channel",
    "payment_mode",
    "is_gift",
    "family",
    "city",
    "shield_member",
    "month",
]

NUM_FEATURES = [
    "discount_pct",
    "repaired_order_value_inr",
    "promised_delivery_days",
    "customer_prior_orders",
    "customer_prior_returns",
    "prior_return_rate_smoothed",
    "list_price_inr",
    "warranty_months",
    "product_age_days",
]

MODEL_FEATURES = CAT_FEATURES + NUM_FEATURES

# These fields are accepted by the service so the endpoint can receive the source
# snapshot schema, but they are deliberately not model features because they are
# not consistently available at the intended pre-dispatch decision point.
IGNORED_DECISION_TIME_FIELDS = {"last_service_event_type", "pickup_scheduled_at", "source"}


def _record_to_frame(record: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([dict(record)])


def engineer_records(
    records: pd.DataFrame,
    customer_ref: Mapping[str, Mapping[str, Any]],
    product_ref: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Create the pre-dispatch feature matrix used by the model."""
    df = records.copy()
    if "order_placed_at" not in df.columns:
        raise ValueError("order_placed_at is required")
    df["order_placed_at"] = pd.to_datetime(df["order_placed_at"], errors="raise")

    customer_rows = []
    product_rows = []
    for _, row in df.iterrows():
        cid = str(row["customer_id"])
        sku = str(row["sku"])
        if cid not in customer_ref:
            raise ValueError(f"Unknown customer_id: {cid}")
        if sku not in product_ref:
            raise ValueError(f"Unknown sku: {sku}")
        customer_rows.append(customer_ref[cid])
        product_rows.append(product_ref[sku])

    cdf = pd.DataFrame(customer_rows).reset_index(drop=True)
    pdf = pd.DataFrame(product_rows).reset_index(drop=True)
    df = df.reset_index(drop=True)

    df["city"] = cdf["city"].astype(str)
    df["state"] = cdf["state"].astype(str)
    df["shield_member"] = cdf["shield_member"].astype(str)
    df["family"] = pdf["family"].astype(str)
    df["list_price_inr"] = pd.to_numeric(pdf["list_price_inr"], errors="raise")
    df["warranty_months"] = pd.to_numeric(pdf["warranty_months"], errors="raise")
    launch = pd.to_datetime(pdf["launch_date"], errors="raise")

    # Do not rely directly on the payment-system amount. Reconstruct the checkout
    # amount from stable order/product fields so known source-system anomalies in
    # the authorized private data cannot dominate the model.
    df["repaired_order_value_inr"] = (
        df["list_price_inr"]
        * pd.to_numeric(df["qty"], errors="raise")
        * (1 - pd.to_numeric(df["discount_pct"], errors="raise") / 100.0)
    )

    prior_orders = pd.to_numeric(df["customer_prior_orders"], errors="raise")
    prior_returns = pd.to_numeric(df["customer_prior_returns"], errors="raise")
    df["prior_return_rate_smoothed"] = (prior_returns + 0.5) / (prior_orders + 5.0)

    df["product_age_days"] = (df["order_placed_at"] - launch).dt.days.clip(lower=0)
    df["month"] = df["order_placed_at"].dt.month.astype(str)

    for col in CAT_FEATURES:
        df[col] = df[col].astype(str)
    for col in NUM_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df[MODEL_FEATURES]


def engineer_one(
    record: Mapping[str, Any],
    customer_ref: Mapping[str, Mapping[str, Any]],
    product_ref: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    return engineer_records(_record_to_frame(record), customer_ref, product_ref)
