from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import CAT_FEATURES, MODEL_FEATURES, NUM_FEATURES, engineer_records


def refs(customers: pd.DataFrame, products: pd.DataFrame) -> tuple[dict, dict]:
    customer_ref = {
        str(r.customer_id): {
            "city": str(r.city),
            "state": str(r.state),
            "shield_member": str(r.shield_member),
        }
        for r in customers.itertuples(index=False)
    }
    product_ref = {
        str(r.sku): {
            "family": str(r.family),
            "list_price_inr": float(r.list_price_inr),
            "warranty_months": int(r.warranty_months),
            "launch_date": str(r.launch_date),
        }
        for r in products.itertuples(index=False)
    }
    return customer_ref, product_ref


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
            ("clf", LogisticRegression(C=0.05, max_iter=3000, random_state=42)),
        ]
    )


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p >= 0.5
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "accuracy_at_0_5": float(accuracy_score(y, pred)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Authorized-data training path. Outputs are private and gitignored. "
            "Supply policy values from the assessment pack at run time; they are not published here."
        )
    )
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "private_output")
    ap.add_argument("--return-cost", type=float, required=True)
    ap.add_argument("--intervention-cost", type=float, required=True)
    ap.add_argument("--intervention-effectiveness", type=float, required=True)
    args = ap.parse_args()

    if args.return_cost <= 0 or args.intervention_cost <= 0:
        raise ValueError("Costs must be positive")
    if not 0 < args.intervention_effectiveness <= 1:
        raise ValueError("intervention-effectiveness must be in (0, 1]")

    inp = args.input_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(inp / "train.csv")
    test = pd.read_csv(inp / "test_unlabelled.csv")
    customers = pd.read_csv(inp / "customers.csv")
    products = pd.read_csv(inp / "products.csv")
    sample = pd.read_csv(inp / "sample_submission.csv")

    # One model row per order. The source-system marker is not used as a feature.
    train = train.drop_duplicates(subset=["order_id"], keep="first").copy().reset_index(drop=True)
    train["order_dt"] = pd.to_datetime(train["order_placed_at"], errors="raise")
    test_dt = pd.to_datetime(test["order_placed_at"], errors="raise")

    customer_ref, product_ref = refs(customers, products)
    X = engineer_records(train.drop(columns=["returned"]), customer_ref, product_ref)
    y = train["returned"].astype(int).to_numpy()
    X_test = engineer_records(test, customer_ref, product_ref)

    # Hidden set is the newest period, so validation must move forward in time.
    hidden_start = test_dt.min().normalize()
    validation_start = hidden_start - pd.DateOffset(months=3)
    train_mask = train["order_dt"] < validation_start
    val_mask = (train["order_dt"] >= validation_start) & (train["order_dt"] < hidden_start)
    if int(val_mask.sum()) == 0 or int(train_mask.sum()) == 0:
        raise RuntimeError("Could not create the intended chronological validation split")

    model = build_model()
    model.fit(X.loc[train_mask.values, MODEL_FEATURES], y[train_mask.to_numpy()])
    p_val = model.predict_proba(X.loc[val_mask.values, MODEL_FEATURES])[:, 1]
    y_val = y[val_mask.to_numpy()]

    threshold = args.intervention_cost / (args.return_cost * args.intervention_effectiveness)
    op = p_val >= threshold
    tp = int(((y_val == 1) & op).sum())
    called = int(op.sum())
    net = (
        tp * args.intervention_effectiveness * args.return_cost
        - called * args.intervention_cost
    )

    validation = evaluate(y_val, p_val)
    validation.update(
        {
            "validation_start": str(validation_start.date()),
            "validation_end_exclusive": str(hidden_start.date()),
            "operational_threshold": float(threshold),
            "intervention_share": float(op.mean()),
            "return_recall_at_operational_threshold": float(tp / max(1, int((y_val == 1).sum()))),
            "estimated_net_benefit_on_validation": float(net),
        }
    )
    (out / "validation_metrics.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    # Fit final model on all authorized historical rows.
    final_model = build_model()
    final_model.fit(X[MODEL_FEATURES], y)
    p_test = final_model.predict_proba(X_test[MODEL_FEATURES])[:, 1]

    score_map = dict(zip(test["order_id"].astype(str), p_test))
    predictions = sample[["order_id"]].copy()
    predictions["score"] = predictions["order_id"].astype(str).map(score_map)
    if predictions["score"].isna().any():
        raise RuntimeError("At least one sample_submission order_id did not receive a score")
    predictions.to_csv(out / "predictions.csv", index=False, float_format="%.8f")

    bundle = {
        "model": final_model,
        "customer_ref": customer_ref,
        "product_ref": product_ref,
        "call_threshold": float(threshold),
        "metadata": {
            "demo_only": False,
            "trained_rows": int(len(train)),
            "features": MODEL_FEATURES,
            "ignored_snapshot_fields": ["last_service_event_type", "pickup_scheduled_at", "source"],
        },
    }
    joblib.dump(bundle, out / "model_bundle.joblib", compress=3)

    print(json.dumps(validation, indent=2))
    print(f"Private artifacts written to: {out}")


if __name__ == "__main__":
    main()
