from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _friendly(feature_name: str, row: pd.Series) -> str | None:
    # Public-safe wording: these describe the fitted contribution for this score
    # without publishing private cohort statistics.
    if feature_name.endswith("payment_mode_cod"):
        return "Cash-on-delivery payment contributes positively to this risk score."
    if feature_name.endswith("shield_member_Y"):
        return "Membership status contributes positively to this fitted risk score."
    if feature_name.endswith("sales_channel_marketplace"):
        return "The marketplace channel contributes positively to this risk score."
    if feature_name.endswith("is_gift_Y"):
        return "Gift status contributes positively to this risk score."
    if "cat__family_" in feature_name:
        family = feature_name.split("cat__family_", 1)[1]
        return f"The product family ({family}) contributes positively to this score."
    if "cat__city_" in feature_name:
        city = feature_name.split("cat__city_", 1)[1]
        return f"The delivery city ({city}) contributes positively to this score."

    if feature_name.endswith("promised_delivery_days"):
        return "The promised delivery window contributes positively to this fitted risk score."
    if feature_name.endswith("customer_prior_returns"):
        if float(row.get("customer_prior_returns", 0)) > 0:
            return "The customer has prior returns, which raises the fitted risk score."
        return None
    if feature_name.endswith("prior_return_rate_smoothed"):
        if float(row.get("customer_prior_returns", 0)) > 0:
            return "The customer's smoothed prior-return history raises the fitted risk score."
        return None
    if feature_name.endswith("discount_pct"):
        return "The checkout discount contributes positively to this fitted risk score."
    if feature_name.endswith("list_price_inr"):
        return "The product list price contributes positively to this fitted risk score."
    if feature_name.endswith("repaired_order_value_inr"):
        return "The reconstructed checkout value contributes positively to this fitted risk score."
    if feature_name.endswith("customer_prior_orders"):
        return "The customer's prior order history contributes positively to this fitted risk score."
    if feature_name.endswith("product_age_days"):
        return "Product age contributes positively to this fitted risk score."
    return None


def reasons_from_linear_model(
    pipeline: Any,
    engineered_row: pd.DataFrame,
    raw_record: dict[str, Any],
    limit: int = 3,
) -> list[str]:
    """Return human-readable reasons from positive per-feature log-odds contributions."""
    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]
    transformed = pre.transform(engineered_row)
    if hasattr(transformed, "toarray"):
        values = transformed.toarray()[0]
    else:
        values = np.asarray(transformed)[0]
    names = pre.get_feature_names_out()
    contributions = values * clf.coef_[0]
    order = np.argsort(contributions)[::-1]

    raw = pd.Series(raw_record)
    result: list[str] = []
    seen: set[str] = set()
    for idx in order:
        if contributions[idx] <= 0:
            break
        msg = _friendly(str(names[idx]), raw)
        if msg and msg not in seen:
            result.append(msg)
            seen.add(msg)
        if len(result) >= limit:
            break

    if not result:
        result.append(
            "No single risk factor dominates; the score comes from the combined order, customer and product profile."
        )
    return result
