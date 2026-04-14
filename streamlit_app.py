from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from gsk_model import (
    build_modeling_frame,
    evaluate_predictions,
    fit_alr_models,
    fit_dirichlet_model,
    fit_random_forest_models,
    fit_xgboost_models,
    get_feature_frame,
    get_target_frame,
    load_exported_artifacts,
    predict_alr,
    predict_dirichlet,
    predict_multinomial,
    predict_random_forest,
    predict_xgboost,
    prepare_app_features_from_inputs,
    train_and_export,
    FEATURE_COLUMNS,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAGE_TITLE = "GSK · Sales by Indication"

MODEL_OPTIONS: dict[str, str] = {
    "Multinomial Logistic Regression": "multinomial",
    "ALR OLS Benchmark": "alr",
    "Dirichlet Regression": "dirichlet",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
}

MODEL_DESCRIPTIONS: dict[str, str] = {
    "multinomial": "Champion model. Weighted multinomial logistic regression trained on sales-weighted hospital observations. Naturally produces a valid 3-way probability split and remains highly interpretable.",
    "alr": "Statistical benchmark. Additive log-ratio OLS regression — a classic compositional data approach used as a performance baseline.",
    "dirichlet": "Probabilistic model. Custom Dirichlet regression maximises a likelihood designed for compositional outputs, offering calibrated uncertainty.",
    "random_forest": "Ensemble model. Three independent Random Forest regressors with simplex normalisation — captures non-linear interactions between touchpoints and HCP reach.",
    "xgboost": "Gradient boosting model. Three XGBoost regressors with simplex normalisation — high flexibility with regularisation to avoid overfitting.",
}

MODEL_TAGS: dict[str, str] = {
    "multinomial":   "Interpretable",
    "alr":           "Baseline",
    "dirichlet":     "Probabilistic",
    "random_forest": "Ensemble",
    "xgboost":       "Boosting",
}

INDICATION_COLORS = {
    "A": "#1E257F",
    "B": "#FF6A00",
    "C": "#2AA198",
}

ORANGE = "#FF6A00"
NAVY  = "#1E257F"

FEATURE_LABELS: dict[str, str] = {
    "log_total_6m_sales": "Log Total Sales",
    "sales_cv": "Sales Variability (CV)",
    "total_touchpoints_all": "Total Touchpoints",
    "touchpoints_share_a": "Touchpoints Share A",
    "touchpoints_share_b": "Touchpoints Share B",
    "total_hcps_all": "Total HCPs",
    "hcp_share_a": "HCP Share A",
    "hcp_share_b": "HCP Share B",
    "tp_per_hcp_a": "Touchpoints/HCP A",
    "tp_per_hcp_b": "Touchpoints/HCP B",
    "tp_per_hcp_c": "Touchpoints/HCP C",
}

PAGES = ["Overview", "Model Comparison", "Calculator", "Data Explorer"]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🟠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

def add_app_style() -> None:
    st.markdown(
        """
        <style>
            /* ---- Base ---- */
            .stApp {
                background:
                    radial-gradient(ellipse at top right, rgba(255,106,0,0.07) 0%, transparent 55%),
                    radial-gradient(ellipse at bottom left, rgba(30,37,127,0.05) 0%, transparent 50%),
                    var(--background-color);
            }
            [data-testid="stHeader"] { background: transparent; }
            [data-testid="stSidebar"] {
                border-right: 1px solid rgba(127,127,127,0.14);
            }

            /* ---- Sidebar brand ---- */
            .sidebar-brand { margin: 0.2rem 0 1.4rem; }
            .sidebar-logo {
                font-size: 2rem;
                font-weight: 900;
                letter-spacing: 0.04em;
                color: #FF6A00;
                margin: 0;
                line-height: 1;
            }
            .sidebar-logo span { color: var(--text-color); font-weight: 400; font-size: 1rem; }
            .sidebar-subtitle {
                margin: 0.25rem 0 0;
                color: rgba(127,127,127,0.9);
                font-size: 0.92rem;
                line-height: 1.35;
            }
            .sidebar-section {
                font-size: 0.75rem;
                color: rgba(127,127,127,0.85);
                text-transform: uppercase;
                letter-spacing: 0.09em;
                margin: 1.2rem 0 0.4rem;
                font-weight: 700;
            }
            .sidebar-divider {
                border: none;
                border-top: 1px solid rgba(127,127,127,0.15);
                margin: 0.8rem 0;
            }

            /* ---- Page titles ---- */
            .page-title {
                font-size: 2.1rem;
                font-weight: 800;
                letter-spacing: -0.03em;
                margin: 0 0 0.2rem;
                line-height: 1.1;
            }
            .page-subtitle {
                color: rgba(127,127,127,0.9);
                margin: 0 0 1.4rem;
                font-size: 1rem;
                line-height: 1.5;
            }
            .section-title {
                font-size: 1.15rem;
                font-weight: 700;
                margin: 1.6rem 0 0.6rem;
                letter-spacing: -0.01em;
            }

            /* ---- Cards ---- */
            .card {
                background: var(--secondary-background-color);
                border: 1px solid rgba(127,127,127,0.16);
                border-radius: 18px;
                padding: 1.1rem 1.2rem;
                height: 100%;
            }
            .card h3 {
                margin: 0 0 0.35rem;
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: #FF6A00;
                font-weight: 700;
            }
            .card p { margin: 0; line-height: 1.5; font-size: 0.95rem; }

            /* ---- KPI tiles ---- */
            .kpi-tile {
                background: var(--secondary-background-color);
                border: 1px solid rgba(127,127,127,0.16);
                border-radius: 16px;
                padding: 1rem 1.1rem;
                text-align: center;
            }
            .kpi-tile .kpi-label {
                font-size: 0.73rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: rgba(127,127,127,0.85);
                font-weight: 700;
                margin-bottom: 0.25rem;
            }
            .kpi-tile .kpi-value {
                font-size: 1.75rem;
                font-weight: 800;
                color: #FF6A00;
                line-height: 1.1;
            }
            .kpi-tile .kpi-sub {
                font-size: 0.78rem;
                color: rgba(127,127,127,0.75);
                margin-top: 0.15rem;
            }

            /* ---- Chips ---- */
            .chip-wrap { display: flex; gap: 0.55rem; flex-wrap: wrap; margin-bottom: 1.1rem; }
            .chip {
                background: rgba(255,106,0,0.08);
                border: 1px solid rgba(255,106,0,0.22);
                border-radius: 999px;
                padding: 0.4rem 0.85rem;
                font-size: 0.85rem;
                font-weight: 600;
                color: #FF6A00;
            }
            .chip-navy {
                background: rgba(30,37,127,0.08);
                border: 1px solid rgba(30,37,127,0.18);
                color: #1E257F;
            }
            .chip-green {
                background: rgba(42,161,152,0.08);
                border: 1px solid rgba(42,161,152,0.22);
                color: #2AA198;
            }

            /* ---- Note banner ---- */
            .note-banner {
                background: rgba(255,106,0,0.07);
                border: 1px solid rgba(255,106,0,0.2);
                border-left: 4px solid #FF6A00;
                border-radius: 10px;
                padding: 0.75rem 1rem;
                margin: 0.6rem 0 1rem;
                font-size: 0.92rem;
                line-height: 1.5;
            }
            .note-banner.navy {
                background: rgba(30,37,127,0.06);
                border-color: rgba(30,37,127,0.18);
                border-left-color: #1E257F;
            }
            .note-banner.green {
                background: rgba(42,161,152,0.07);
                border-color: rgba(42,161,152,0.22);
                border-left-color: #2AA198;
            }

            /* ---- Model card ---- */
            .model-card {
                background: var(--secondary-background-color);
                border: 1.5px solid rgba(255,106,0,0.2);
                border-radius: 16px;
                padding: 1rem 1.1rem;
            }
            .model-card .model-tag {
                display: inline-block;
                background: rgba(255,106,0,0.1);
                border: 1px solid rgba(255,106,0,0.25);
                border-radius: 999px;
                padding: 0.18rem 0.6rem;
                font-size: 0.72rem;
                font-weight: 700;
                color: #FF6A00;
                letter-spacing: 0.06em;
                margin-bottom: 0.4rem;
            }
            .model-card .model-name {
                font-size: 1.05rem;
                font-weight: 700;
                margin: 0 0 0.3rem;
            }
            .model-card .model-desc {
                font-size: 0.88rem;
                color: rgba(127,127,127,0.9);
                line-height: 1.5;
                margin: 0;
            }

            /* ---- Prediction result bar ---- */
            .pred-row {
                display: flex;
                align-items: center;
                gap: 0.8rem;
                margin-bottom: 0.7rem;
            }
            .pred-label {
                font-weight: 700;
                font-size: 0.95rem;
                min-width: 5.5rem;
            }
            .pred-bar-wrap {
                flex: 1;
                background: rgba(127,127,127,0.1);
                border-radius: 999px;
                height: 22px;
                overflow: hidden;
            }
            .pred-bar-fill {
                height: 100%;
                border-radius: 999px;
                transition: width 0.4s ease;
            }
            .pred-pct {
                font-weight: 700;
                font-size: 1rem;
                min-width: 3.5rem;
                text-align: right;
            }

            /* ---- Table ---- */
            .styled-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.9rem;
                margin-top: 0.5rem;
            }
            .styled-table th {
                text-align: left;
                padding: 0.5rem 0.7rem;
                font-size: 0.73rem;
                text-transform: uppercase;
                letter-spacing: 0.07em;
                color: rgba(127,127,127,0.8);
                border-bottom: 1px solid rgba(127,127,127,0.18);
            }
            .styled-table td {
                padding: 0.45rem 0.7rem;
                border-bottom: 1px solid rgba(127,127,127,0.09);
            }
            .styled-table tr:last-child td { border-bottom: none; }
            .styled-table .best { font-weight: 700; color: #FF6A00; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data loading & caching
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Training models — this takes ~30 s on first load…")
def get_demo_bundle() -> dict[str, Any]:
    train_and_export()
    multinomial_model, config = load_exported_artifacts()
    modeling_df = build_modeling_frame(config["data_path"])

    X_full = get_feature_frame(modeling_df)
    y_full = get_target_frame(modeling_df)

    alr_models      = fit_alr_models(X_full, y_full)
    dirichlet_model = fit_dirichlet_model(X_full, y_full)
    rf_models       = fit_random_forest_models(X_full, y_full)
    xgb_models      = fit_xgboost_models(X_full, y_full)

    # Quick hold-out metrics for all 5 models (80/20 split)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(modeling_df, test_size=0.2, random_state=42)
    X_tr = get_feature_frame(train_df); y_tr = get_target_frame(train_df)
    X_te = get_feature_frame(test_df);  y_te = get_target_frame(test_df)

    alr_te  = fit_alr_models(X_tr, y_tr)
    dir_te  = fit_dirichlet_model(X_tr, y_tr)
    rf_te   = fit_random_forest_models(X_tr, y_tr)
    xgb_te  = fit_xgboost_models(X_tr, y_tr)
    from gsk_model import fit_weighted_multinomial
    mn_te   = fit_weighted_multinomial(X_tr, y_tr, train_df["total_6m_sales"], C=config["best_c"])

    all_metrics: dict[str, pd.DataFrame] = {}
    for name, preds in [
        ("Multinomial LR",  predict_multinomial(mn_te, X_te)),
        ("ALR Benchmark",   predict_alr(alr_te, X_te)),
        ("Dirichlet",       predict_dirichlet(dir_te, X_te)),
        ("Random Forest",   predict_random_forest(rf_te, X_te)),
        ("XGBoost",         predict_xgboost(xgb_te, X_te)),
    ]:
        all_metrics[name] = evaluate_predictions(y_te, preds, dataset_label="holdout")

    default_raw_inputs = {
        "total_6m_sales": int(round(modeling_df["total_6m_sales"].median())),
        "touchpoints_a":  int(round(modeling_df["total_touchpoints_a"].median())),
        "touchpoints_b":  int(round(modeling_df["total_touchpoints_b"].median())),
        "touchpoints_c":  int(round(modeling_df["total_touchpoints_c"].median())),
        "hcps_a":         int(round(modeling_df["total_hcps_a"].median())),
        "hcps_b":         int(round(modeling_df["total_hcps_b"].median())),
        "hcps_c":         int(round(modeling_df["total_hcps_c"].median())),
    }

    return {
        "multinomial_model": multinomial_model,
        "alr_models":        alr_models,
        "dirichlet_model":   dirichlet_model,
        "rf_models":         rf_models,
        "xgb_models":        xgb_models,
        "config":            config,
        "modeling_df":       modeling_df,
        "all_metrics":       all_metrics,
        "sales_cv_default":  float(config["feature_ranges"]["sales_cv"]["median"]),
        "default_raw_inputs": default_raw_inputs,
    }


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def build_model_inputs(raw: dict[str, float], sales_cv_default: float) -> pd.DataFrame:
    tp_total  = raw["touchpoints_a"] + raw["touchpoints_b"] + raw["touchpoints_c"]
    hcp_total = raw["hcps_a"] + raw["hcps_b"] + raw["hcps_c"]
    return prepare_app_features_from_inputs({
        "total_6m_sales":     raw["total_6m_sales"],
        "sales_cv":           sales_cv_default,
        "total_touchpoints_all": tp_total,
        "touchpoints_share_a":   raw["touchpoints_a"] / (tp_total  + 1e-6),
        "touchpoints_share_b":   raw["touchpoints_b"] / (tp_total  + 1e-6),
        "total_hcps_all":        hcp_total,
        "hcp_share_a":           raw["hcps_a"] / (hcp_total + 1e-6),
        "hcp_share_b":           raw["hcps_b"] / (hcp_total + 1e-6),
    })


def predict_scenario(bundle: dict[str, Any], raw: dict[str, float], model_key: str) -> pd.Series:
    features = build_model_inputs(raw, bundle["sales_cv_default"])
    dispatch = {
        "multinomial":  lambda: predict_multinomial(bundle["multinomial_model"], features),
        "alr":          lambda: predict_alr(bundle["alr_models"], features),
        "dirichlet":    lambda: predict_dirichlet(bundle["dirichlet_model"], features),
        "random_forest":lambda: predict_random_forest(bundle["rf_models"], features),
        "xgboost":      lambda: predict_xgboost(bundle["xgb_models"], features),
    }
    return dispatch[model_key]().iloc[0]


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _theme(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(size=13),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(zeroline=False, showgrid=True, gridcolor="rgba(127,127,127,0.1)")
    fig.update_yaxes(zeroline=False, showgrid=True, gridcolor="rgba(127,127,127,0.1)")
    return fig


def chart_units(prediction: pd.Series, total_sales: float) -> go.Figure:
    ind   = ["A", "B", "C"]
    units = [prediction[f"pred_split_{k.lower()}"] * total_sales for k in ind]
    colors = [INDICATION_COLORS[k] for k in ind]
    fig = go.Figure(go.Bar(
        x=[f"Indication {k}" for k in ind],
        y=units,
        marker_color=colors,
        text=[f"{u:,.0f}" for u in units],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y:,.0f} units<extra></extra>",
    ))
    fig.update_layout(title="Predicted units by indication", yaxis_title="Units")
    return _theme(fig, 370)


def chart_mix(prediction: pd.Series) -> go.Figure:
    vals = [prediction[f"pred_split_{k.lower()}"] * 100 for k in ["A", "B", "C"]]
    fig = go.Figure(go.Pie(
        labels=[f"Indication {k}" for k in ["A", "B", "C"]],
        values=vals,
        hole=0.56,
        marker=dict(colors=[INDICATION_COLORS[k] for k in ["A", "B", "C"]]),
        textinfo="label+percent",
        hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>",
        sort=False,
    ))
    fig.update_layout(title="Predicted indication mix", showlegend=False)
    return _theme(fig, 340)


def chart_all_models_comparison(bundle: dict[str, Any], raw: dict[str, float]) -> go.Figure:
    """Grouped bar: predicted shares for all 5 models side by side."""
    model_labels = list(MODEL_OPTIONS.keys())
    model_keys   = list(MODEL_OPTIONS.values())
    indications  = ["A", "B", "C"]

    rows = []
    for label, key in zip(model_labels, model_keys):
        pred = predict_scenario(bundle, raw, key)
        for ind in indications:
            rows.append({
                "Model": label.replace(" ", "<br>"),
                "Indication": f"Indication {ind}",
                "Share (%)": pred[f"pred_split_{ind.lower()}"] * 100,
            })
    df = pd.DataFrame(rows)

    fig = go.Figure()
    for ind in indications:
        sub = df[df["Indication"] == f"Indication {ind}"]
        fig.add_trace(go.Bar(
            name=f"Indication {ind}",
            x=sub["Model"],
            y=sub["Share (%)"],
            marker_color=INDICATION_COLORS[ind],
            text=[f"{v:.1f}%" for v in sub["Share (%)"]],
            textposition="outside",
            hovertemplate=f"<b>Indication {ind}</b><br>%{{x}}<br>%{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        title="All models — predicted indication split",
        barmode="group",
        yaxis_title="Share (%)",
        yaxis=dict(range=[0, 80]),
    )
    return _theme(fig, 420)


def chart_mae_heatmap(all_metrics: dict[str, pd.DataFrame]) -> go.Figure:
    """MAE heatmap: models × indications."""
    model_names = list(all_metrics.keys())
    indications = ["A", "B", "C"]
    z = []
    for model in model_names:
        m_df = all_metrics[model]
        row = []
        for ind in indications:
            sub = m_df[m_df["indication"] == ind]
            row.append(float(sub["mae"].values[0]) if len(sub) else float("nan"))
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"Ind. {i}" for i in indications],
        y=model_names,
        colorscale=[[0, "#FF6A00"], [0.5, "#FFC080"], [1, "#F5F5F5"]],
        reversescale=True,
        text=[[f"{v:.4f}" for v in row] for row in z],
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b><br>Indication %{x}<br>MAE: %{z:.4f}<extra></extra>",
        showscale=True,
        colorbar=dict(title="MAE", thickness=12),
    ))
    fig.update_layout(title="Mean Absolute Error — holdout (lower = better)")
    return _theme(fig, 340)


def chart_mae_bar(all_metrics: dict[str, pd.DataFrame]) -> go.Figure:
    """Average MAE across indications per model."""
    model_names, avg_maes = [], []
    for name, m_df in all_metrics.items():
        avg_maes.append(m_df["mae"].mean())
        model_names.append(name)
    best_idx = int(np.argmin(avg_maes))
    colors = [ORANGE if i == best_idx else "#CBD5E1" for i in range(len(model_names))]
    fig = go.Figure(go.Bar(
        x=model_names,
        y=avg_maes,
        marker_color=colors,
        text=[f"{v:.4f}" for v in avg_maes],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Avg MAE: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Average MAE across indications — holdout",
        yaxis_title="MAE",
        yaxis=dict(range=[0, max(avg_maes) * 1.25]),
    )
    return _theme(fig, 350)


def chart_rmse_bar(all_metrics: dict[str, pd.DataFrame]) -> go.Figure:
    """Average RMSE across indications per model."""
    model_names, avg_rmses = [], []
    for name, m_df in all_metrics.items():
        avg_rmses.append(m_df["rmse"].mean())
        model_names.append(name)
    best_idx = int(np.argmin(avg_rmses))
    colors = [ORANGE if i == best_idx else "#CBD5E1" for i in range(len(model_names))]
    fig = go.Figure(go.Bar(
        x=model_names,
        y=avg_rmses,
        marker_color=colors,
        text=[f"{v:.4f}" for v in avg_rmses],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Avg RMSE: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Average RMSE across indications — holdout",
        yaxis_title="RMSE",
        yaxis=dict(range=[0, max(avg_rmses) * 1.25]),
    )
    return _theme(fig, 350)


def chart_radar(all_metrics: dict[str, pd.DataFrame]) -> go.Figure:
    categories = ["Indication A", "Indication B", "Indication C"]
    fig = go.Figure()
    palette = [ORANGE, NAVY, "#2AA198", "#9B59B6", "#E74C3C"]

    # Global max MAE for consistent scaling
    global_max = max(
        float(m_df["mae"].max())
        for m_df in all_metrics.values()
    )

    for (name, m_df), color in zip(all_metrics.items(), palette):
        values = []
        for ind in ["A", "B", "C"]:
            sub = m_df[m_df["indication"] == ind]
            values.append(float(sub["mae"].values[0]) if len(sub) else 0.0)
        # Invert so higher = better
        inv = [1 - v / (global_max + 1e-6) for v in values]
        inv_closed = inv + [inv[0]]
        cats_closed = categories + [categories[0]]
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatterpolar(
            r=inv_closed,
            theta=cats_closed,
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.15)",
            line=dict(color=color, width=2),
            name=name,
        ))
    fig.update_layout(
        title="Model accuracy profile (higher = better per axis)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
    )
    return _theme(fig, 420)


def chart_feature_importance(config: dict[str, Any]) -> go.Figure:
    """Coefficient heatmap for champion multinomial model."""
    top_drivers = config.get("top_drivers", {})
    if not top_drivers:
        return go.Figure()

    indications = ["A", "B", "C"]
    all_feats = FEATURE_COLUMNS
    coefs: dict[str, dict[str, float]] = {ind: {} for ind in indications}

    for ind in indications:
        pos = top_drivers.get(ind, {}).get("positive", {})
        neg = top_drivers.get(ind, {}).get("negative", {})
        combined = {**pos, **neg}
        for feat in all_feats:
            coefs[ind][feat] = combined.get(feat, 0.0)

    z = [[coefs[ind].get(feat, 0.0) for ind in indications] for feat in all_feats]
    y_labels = [FEATURE_LABELS.get(f, f) for f in all_feats]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"Indication {i}" for i in indications],
        y=y_labels,
        colorscale="RdBu",
        zmid=0,
        text=[[f"{v:.3f}" for v in row] for row in z],
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b><br>%{x}<br>Coefficient: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Coef.", thickness=12),
    ))
    fig.update_layout(title="Champion model — feature coefficients by indication")
    return _theme(fig, 460)


def chart_sales_distribution(modeling_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=modeling_df["total_6m_sales"],
        nbinsx=30,
        marker_color=ORANGE,
        opacity=0.8,
        hovertemplate="Sales: %{x}<br>Hospitals: %{y}<extra></extra>",
    ))
    fig.update_layout(title="Distribution of 6-month hospital sales", xaxis_title="Units", yaxis_title="# Hospitals")
    return _theme(fig, 320)


def chart_touchpoints_vs_split(modeling_df: pd.DataFrame, indication: str = "a") -> go.Figure:
    ind_up = indication.upper()
    col_tp  = f"touchpoints_share_{indication}"
    col_sp  = f"avg_split_{indication}"
    has_label = modeling_df[col_sp].notna()
    df = modeling_df[has_label].copy()
    fig = px.scatter(
        df, x=col_tp, y=col_sp,
        color_discrete_sequence=[INDICATION_COLORS[ind_up]],
        labels={col_tp: f"Touchpoints Share — Ind. {ind_up}", col_sp: f"Actual Sales Share — Ind. {ind_up}"},
    )
    fig.update_traces(marker=dict(size=7, opacity=0.65))
    fig.update_layout(title=f"Touchpoints share vs actual sales split — Indication {ind_up}")
    return _theme(fig, 340)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def initialize_state(defaults: dict[str, int]) -> None:
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if "submitted_raw_inputs" not in st.session_state:
        st.session_state["submitted_raw_inputs"] = {k: float(v) for k, v in defaults.items()}
    if "submitted_model_key" not in st.session_state:
        st.session_state["submitted_model_key"] = "multinomial"
    if "page" not in st.session_state:
        st.session_state["page"] = PAGES[0]


def render_sidebar(bundle: dict[str, Any]) -> tuple[str, str]:
    initialize_state(bundle["default_raw_inputs"])

    with st.sidebar:
        # Brand
        st.markdown(
            """
            <div class="sidebar-brand">
                <p class="sidebar-logo">GSK<span> · Sales Analytics</span></p>
                <p class="sidebar-subtitle">Hospital Indication Split Model</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Navigation
        st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
        page = st.radio(
            "Page",
            PAGES,
            index=PAGES.index(st.session_state.get("page", PAGES[0])),
            label_visibility="collapsed",
        )
        st.session_state["page"] = page

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # Model selector
        st.markdown('<div class="sidebar-section">Active Model</div>', unsafe_allow_html=True)
        model_label = st.selectbox(
            "Model",
            list(MODEL_OPTIONS.keys()),
            label_visibility="collapsed",
        )
        active_model_key = MODEL_OPTIONS[model_label]

        # Model description pill
        tag = MODEL_TAGS[active_model_key]
        desc = MODEL_DESCRIPTIONS[active_model_key]
        key_to_metric = {
            "multinomial":   "Multinomial LR",
            "alr":           "ALR Benchmark",
            "dirichlet":     "Dirichlet",
            "random_forest": "Random Forest",
            "xgboost":       "XGBoost",
        }
        best_key = min(
            key_to_metric.keys(),
            key=lambda k: bundle["all_metrics"][key_to_metric[k]]["mae"].mean()
        )
        if active_model_key == best_key:
            tag = "⭐ Best Model"
        tag_color = "chip" if active_model_key == best_key else "chip chip-navy"
        st.markdown(
            f"""
            <div style="margin-top:0.4rem;">
                <div class="{tag_color}" style="display:inline-block;margin-bottom:0.45rem;">{tag}</div>
                <p style="font-size:0.83rem;color:rgba(127,127,127,0.9);line-height:1.45;margin:0;">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # Hospital inputs
        st.markdown('<div class="sidebar-section">Hospital Inputs</div>', unsafe_allow_html=True)
        with st.form("hospital_form"):
            total_6m_sales = st.number_input(
                "Total 6-month sales (units)",
                min_value=1, step=25,
                value=int(st.session_state["total_6m_sales"]),
                help="Total units purchased by the hospital over 6 months.",
            )
            st.markdown("**Touchpoints by indication**")
            c1, c2, c3 = st.columns(3)
            touchpoints_a = c1.number_input("Ind. A", min_value=0, step=5, value=int(st.session_state["touchpoints_a"]), key="tp_a_inp")
            touchpoints_b = c2.number_input("Ind. B", min_value=0, step=5, value=int(st.session_state["touchpoints_b"]), key="tp_b_inp")
            touchpoints_c = c3.number_input("Ind. C", min_value=0, step=5, value=int(st.session_state["touchpoints_c"]), key="tp_c_inp")

            st.markdown("**HCPs reached by indication**")
            c4, c5, c6 = st.columns(3)
            hcps_a = c4.number_input("Ind. A", min_value=0, step=5, value=int(st.session_state["hcps_a"]), key="hcp_a_inp")
            hcps_b = c5.number_input("Ind. B", min_value=0, step=5, value=int(st.session_state["hcps_b"]), key="hcp_b_inp")
            hcps_c = c6.number_input("Ind. C", min_value=0, step=5, value=int(st.session_state["hcps_c"]), key="hcp_c_inp")

            submitted = st.form_submit_button("▶ Run Prediction", use_container_width=True, type="primary")

        if submitted:
            latest = {
                "total_6m_sales": float(total_6m_sales),
                "touchpoints_a":  float(touchpoints_a),
                "touchpoints_b":  float(touchpoints_b),
                "touchpoints_c":  float(touchpoints_c),
                "hcps_a":         float(hcps_a),
                "hcps_b":         float(hcps_b),
                "hcps_c":         float(hcps_c),
            }
            st.session_state["submitted_raw_inputs"] = latest
            st.session_state["submitted_model_key"]  = active_model_key
            for k, v in latest.items():
                st.session_state[k] = v
            with st.spinner("Scoring scenario…"):
                time.sleep(0.25)

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.75rem;color:rgba(127,127,127,0.6);line-height:1.4;">'
            'Model trained on synthetic GSK hospital sales data. '
            'Best model determined dynamically from holdout evaluation.</p>',
            unsafe_allow_html=True,
        )

    return page, active_model_key


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------

def render_overview(bundle: dict[str, Any]) -> None:
    config      = bundle["config"]
    modeling_df = bundle["modeling_df"]
    all_metrics = bundle["all_metrics"]

    st.markdown('<h1 style="font-size:2.5rem;font-weight:900;letter-spacing:-0.03em;margin-bottom:0.2rem;">Hospital Sales by Indication</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Estimate how total hospital drug sales split across Indications A, B, and C '
        'using commercial activity signals and HCP reach data.</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="chip-wrap">
            <div class="chip">5 Models Evaluated</div>
            <div class="chip chip-navy">100 Hospitals · All Labeled</div>
            <div class="chip chip-green">Shares always sum to 100 %</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI row
    n_hospitals = len(modeling_df)
    n_labeled   = modeling_df["avg_split_a"].notna().sum()
    champion_avg_mae = all_metrics["Multinomial LR"]["mae"].mean()
    total_units = int(modeling_df["total_6m_sales"].sum())

    k1, k2, k3, k4 = st.columns(4)
    for col, label, value, sub in [
        (k1, "Hospitals",         f"{n_hospitals}",           "In dataset"),
        (k2, "Labeled hospitals", f"{n_labeled}",             "With known split"),
        (k3, "Champion MAE",      f"{champion_avg_mae:.4f}",  "Avg across indications"),
        (k4, "Total units (6m)",  f"{total_units:,}",         "Across all hospitals"),
    ]:
        col.markdown(
            f"""<div class="kpi-tile">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-sub">{sub}</div>
               </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Context cards
    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            """<div style="background:rgba(255,106,0,0.06);border:1.5px solid rgba(255,106,0,0.25);
            border-radius:18px;padding:1.4rem 1.5rem;height:100%;">
            <div style="font-size:1.6rem;margin-bottom:0.5rem;">⚠️</div>
            <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.09em;
            color:#FF6A00;margin-bottom:0.6rem;">The Challenge</div>
            <p style="margin:0;line-height:1.6;font-size:0.95rem;">
            Hospitals purchase <strong>one drug prescribed for multiple indications</strong>. 
            Purchase orders do not specify how much was used for <strong>Indication A, B, or C</strong>, 
            making it impossible to assess <strong>commercial effectiveness by disease area</strong>.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            """<div style="background:rgba(30,37,127,0.06);border:1.5px solid rgba(30,37,127,0.2);
            border-radius:18px;padding:1.4rem 1.5rem;height:100%;">
            <div style="font-size:1.6rem;margin-bottom:0.5rem;">✅</div>
            <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.09em;
            color:#1E257F;margin-bottom:0.6rem;">Our Solution</div>
            <p style="margin:0;line-height:1.6;font-size:0.95rem;">
            We model the indication split using <strong>touchpoints and HCP reach</strong> broken down 
            by indication as predictors. <strong>Five models</strong> are compared on a held-out test set. 
            <strong>Indication A dominates</strong> (~66% average share), making B and C harder to predict.</p>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<p class="section-title">Sales distribution across hospitals</p>', unsafe_allow_html=True)
    st.plotly_chart(chart_sales_distribution(modeling_df), use_container_width=True, theme="streamlit")

    st.markdown('<p class="section-title">Touchpoints share vs actual split — Indication A</p>', unsafe_allow_html=True)
    st.plotly_chart(chart_touchpoints_vs_split(modeling_df, "a"), use_container_width=True, theme="streamlit")

    st.markdown('<p class="section-title">Champion model — feature coefficients</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="note-banner">Positive coefficients push allocation toward that indication; '
        'negative coefficients pull away from it. The champion model is fully interpretable by design.</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(chart_feature_importance(config), use_container_width=True, theme="streamlit")

    # Top drivers table
    top_drivers = config.get("top_drivers", {})
    if top_drivers:
        st.markdown('<p class="section-title">Top positive drivers per indication</p>', unsafe_allow_html=True)
        d_cols = st.columns(3)
        for col, ind in zip(d_cols, ["A", "B", "C"]):
            pos = top_drivers.get(ind, {}).get("positive", {})
            rows_html = "".join(
                f"<tr><td>{FEATURE_LABELS.get(f, f)}</td><td class='best'>+{v:.3f}</td></tr>"
                for f, v in list(pos.items())[:4]
            )
            col.markdown(
                f"""<div class="card">
                    <h3>Indication {ind}</h3>
                    <table class="styled-table">{rows_html}</table>
                </div>""",
                unsafe_allow_html=True,
            )


def render_model_comparison(bundle: dict[str, Any]) -> None:
    all_metrics = bundle["all_metrics"]
    raw         = st.session_state["submitted_raw_inputs"]

    st.markdown('<h1 style="font-size:2.5rem;font-weight:900;letter-spacing:-0.03em;margin-bottom:0.2rem;">Model Comparison</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:1.05rem;color:rgba(127,127,127,0.9);margin-bottom:1.4rem;">Head-to-head performance of all five models on the hold-out test set '
        'and side-by-side prediction output for the current hospital scenario.</p>',
        unsafe_allow_html=True,
    )

    # Best model banner
    avg_maes = {name: m_df["mae"].mean() for name, m_df in all_metrics.items()}
    best_name = min(avg_maes, key=avg_maes.get)
    best_mae  = avg_maes[best_name]
    st.markdown(
        f'<div class="note-banner">🏆 <strong>Best model on hold-out: {best_name}</strong> '
        f'— average MAE {best_mae:.4f} across all indications.</div>',
        unsafe_allow_html=True,
    )

    # Metric charts
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_mae_bar(all_metrics), use_container_width=True, theme="streamlit")
    with c2:
        st.plotly_chart(chart_rmse_bar(all_metrics), use_container_width=True, theme="streamlit")

    st.plotly_chart(chart_mae_heatmap(all_metrics), use_container_width=True, theme="streamlit")
    st.plotly_chart(chart_radar(all_metrics), use_container_width=True, theme="streamlit")

    # Per-model cards with MAE + desc
    st.markdown('<p class="section-title">Model details</p>', unsafe_allow_html=True)
    model_items = [
        ("Multinomial LR",  "multinomial",  "Multinomial Logistic Regression"),
        ("ALR Benchmark",   "alr",          "ALR OLS Benchmark"),
        ("Dirichlet",       "dirichlet",    "Dirichlet Regression"),
        ("Random Forest",   "random_forest","Random Forest"),
        ("XGBoost",         "xgboost",      "XGBoost"),
    ]
    for i in range(0, len(model_items), 2):
        pair = model_items[i:i+2]
        cols = st.columns(2)
        for col, (metric_name, key, label) in zip(cols, pair):
            mae  = all_metrics[metric_name]["mae"].mean()
            rmse = all_metrics[metric_name]["rmse"].mean()
            tag  = MODEL_TAGS[key]
            desc = MODEL_DESCRIPTIONS[key]
            is_best = metric_name == best_name
            border_style = "border: 2px solid #FF6A00;" if is_best else ""
            col.markdown(
                f"""<div class="model-card" style="{border_style}">
                    <div class="model-tag">{tag}{' 🏆' if is_best else ''}</div>
                    <div class="model-name">{label}</div>
                    <p class="model-desc">{desc}</p>
                    <div style="display:flex;gap:1.5rem;margin-top:0.7rem;">
                        <div><span style="font-size:0.72rem;color:rgba(127,127,127,0.8);text-transform:uppercase;letter-spacing:.07em;">Avg MAE</span>
                             <br><strong style="color:#FF6A00;">{mae:.4f}</strong></div>
                        <div><span style="font-size:0.72rem;color:rgba(127,127,127,0.8);text-transform:uppercase;letter-spacing:.07em;">Avg RMSE</span>
                             <br><strong style="color:#1E257F;">{rmse:.4f}</strong></div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

    # All-model prediction comparison for current scenario
    st.markdown('<p class="section-title">Predicted splits for current hospital scenario</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="note-banner navy">Adjust the hospital inputs in the sidebar and click <strong>Run Prediction</strong> to update.</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(chart_all_models_comparison(bundle, raw), use_container_width=True, theme="streamlit")

    # Numeric table
    rows = []
    for metric_name, key, label in model_items:
        pred = predict_scenario(bundle, raw, key)
        rows.append({
            "Model": label,
            "Ind. A (%)": f"{pred['pred_split_a']*100:.1f}",
            "Ind. B (%)": f"{pred['pred_split_b']*100:.1f}",
            "Ind. C (%)": f"{pred['pred_split_c']*100:.1f}",
            "Avg MAE": f"{all_metrics[metric_name]['mae'].mean():.4f}",
        })
    header_cols = list(rows[0].keys())
    header_html = "".join(f"<th>{h}</th>" for h in header_cols)
    body_html = ""
    for r in rows:
        cells = "".join(f"<td class='{'best' if r['Model']==best_name.replace('LR','Logistic Regression') else ''}'>{v}</td>" for v in r.values())
        body_html += f"<tr>{cells}</tr>"
    st.markdown(
        f"""<table class="styled-table"><thead><tr>{header_html}</tr></thead>
            <tbody>{body_html}</tbody></table>""",
        unsafe_allow_html=True,
    )


def render_calculator(bundle: dict[str, Any]) -> None:
    raw            = st.session_state["submitted_raw_inputs"]
    model_key      = st.session_state.get("submitted_model_key", "multinomial")
    model_label    = next(lbl for lbl, k in MODEL_OPTIONS.items() if k == model_key)
    prediction     = predict_scenario(bundle, raw, model_key)
    total_sales    = raw["total_6m_sales"]

    st.markdown('<h1 style="font-size:2.5rem;font-weight:900;letter-spacing:-0.03em;margin-bottom:0.2rem;">Prediction Calculator</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Configure a hospital scenario in the sidebar and inspect the '
        'predicted indication split in real time.</p>',
        unsafe_allow_html=True,
    )

    tag   = MODEL_TAGS[model_key]
    desc  = MODEL_DESCRIPTIONS[model_key]
    st.markdown(
        f'<div class="note-banner"><strong>{tag} · {model_label}</strong><br>'
        f'<span style="font-size:0.88rem;">{desc}</span></div>',
        unsafe_allow_html=True,
    )

    # Inline prediction bars
    st.markdown('<p class="section-title">Predicted indication split</p>', unsafe_allow_html=True)
    for ind in ["A", "B", "C"]:
        pct   = prediction[f"pred_split_{ind.lower()}"] * 100
        units = prediction[f"pred_split_{ind.lower()}"] * total_sales
        color = INDICATION_COLORS[ind]
        st.markdown(
            f"""
            <div class="pred-row">
                <div class="pred-label" style="color:{color};">Indication {ind}</div>
                <div class="pred-bar-wrap">
                    <div class="pred-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
                </div>
                <div class="pred-pct" style="color:{color};">{pct:.1f}%</div>
                <div style="font-size:0.85rem;color:rgba(127,127,127,0.75);min-width:6rem;text-align:right;">{units:,.0f} units</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_units(prediction, total_sales), use_container_width=True, theme="streamlit")
    with c2:
        st.plotly_chart(chart_mix(prediction), use_container_width=True, theme="streamlit")

    # Input summary
    st.markdown('<p class="section-title">Scenario summary</p>', unsafe_allow_html=True)
    tp_total  = raw["touchpoints_a"] + raw["touchpoints_b"] + raw["touchpoints_c"]
    hcp_total = raw["hcps_a"] + raw["hcps_b"] + raw["hcps_c"]
    rows_html = "".join([
        f"<tr><td>Total 6m sales</td><td><strong>{int(total_sales):,} units</strong></td></tr>",
        f"<tr><td>Total touchpoints</td><td><strong>{int(tp_total)}</strong></td></tr>",
        f"<tr><td>Total HCPs</td><td><strong>{int(hcp_total)}</strong></td></tr>",
        f"<tr><td>TP share — Ind. A / B / C</td><td><strong>"
        f"{raw['touchpoints_a']/max(tp_total,1)*100:.0f}% / "
        f"{raw['touchpoints_b']/max(tp_total,1)*100:.0f}% / "
        f"{raw['touchpoints_c']/max(tp_total,1)*100:.0f}%</strong></td></tr>",
        f"<tr><td>HCP share — Ind. A / B / C</td><td><strong>"
        f"{raw['hcps_a']/max(hcp_total,1)*100:.0f}% / "
        f"{raw['hcps_b']/max(hcp_total,1)*100:.0f}% / "
        f"{raw['hcps_c']/max(hcp_total,1)*100:.0f}%</strong></td></tr>",
    ])
    st.markdown(
        f"""<table class="styled-table"><tbody>{rows_html}</tbody></table>""",
        unsafe_allow_html=True,
    )

    # Quick compare — show all models for this scenario
    st.markdown('<p class="section-title">Quick comparison across all models</p>', unsafe_allow_html=True)
    model_items = [
        ("Multinomial LR",  "multinomial",  "Multinomial LR"),
        ("ALR Benchmark",   "alr",          "ALR Benchmark"),
        ("Dirichlet",       "dirichlet",    "Dirichlet"),
        ("Random Forest",   "random_forest","Random Forest"),
        ("XGBoost",         "xgboost",      "XGBoost"),
    ]
    all_metrics = bundle["all_metrics"]
    compare_cols = st.columns(5)
    for col, (metric_name, key, short) in zip(compare_cols, model_items):
        pred = predict_scenario(bundle, raw, key)
        is_active = key == model_key
        border = "border:2px solid #FF6A00;" if is_active else ""
        col.markdown(
            f"""<div class="kpi-tile" style="{border}">
                <div class="kpi-label">{short}</div>
                <div style="font-size:0.82rem;margin:0.3rem 0;">
                    <span style="color:{INDICATION_COLORS['A']};font-weight:700;">A {pred['pred_split_a']*100:.0f}%</span><br>
                    <span style="color:{INDICATION_COLORS['B']};font-weight:700;">B {pred['pred_split_b']*100:.0f}%</span><br>
                    <span style="color:{INDICATION_COLORS['C']};font-weight:700;">C {pred['pred_split_c']*100:.0f}%</span>
                </div>
                {"<div class='kpi-sub'>Active</div>" if is_active else ""}
            </div>""",
            unsafe_allow_html=True,
        )


def render_data_explorer(bundle: dict[str, Any]) -> None:
    modeling_df = bundle["modeling_df"]

    st.markdown('<h1 style="font-size:2.5rem;font-weight:900;letter-spacing:-0.03em;margin-bottom:0.2rem;">Data Explorer</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Explore the underlying hospital-level dataset used for training and evaluation.</p>',
        unsafe_allow_html=True,
    )

    n_hosp    = len(modeling_df)
    n_labeled = modeling_df["avg_split_a"].notna().sum()
    median_s  = int(modeling_df["total_6m_sales"].median())
    max_s     = int(modeling_df["total_6m_sales"].max())

    k1, k2, k3, k4 = st.columns(4)
    for col, label, val, sub in [
        (k1, "Total hospitals",    str(n_hosp),    "In dataset"),
        (k2, "Labeled hospitals",  str(n_labeled), "With known split"),
        (k3, "Median sales (6m)",  f"{median_s:,}", "units"),
        (k4, "Max sales (6m)",     f"{max_s:,}",   "units"),
    ]:
        col.markdown(
            f"""<div class="kpi-tile">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<p class="section-title">Sales distribution</p>', unsafe_allow_html=True)
    st.plotly_chart(chart_sales_distribution(modeling_df), use_container_width=True, theme="streamlit")

    st.markdown('<p class="section-title">Touchpoints share vs actual split</p>', unsafe_allow_html=True)
    ind_choice = st.selectbox("Indication", ["A", "B", "C"], index=0)
    st.plotly_chart(
        chart_touchpoints_vs_split(modeling_df, ind_choice.lower()),
        use_container_width=True,
        theme="streamlit",
    )

    # Average split distribution among labeled hospitals
    labeled = modeling_df.dropna(subset=["avg_split_a"]).copy()
    if not labeled.empty:
        st.markdown('<p class="section-title">Actual indication split — labeled hospitals</p>', unsafe_allow_html=True)
        fig = go.Figure()
        for ind in ["A", "B", "C"]:
            fig.add_trace(go.Box(
                y=labeled[f"avg_split_{ind.lower()}"],
                name=f"Indication {ind}",
                marker_color=INDICATION_COLORS[ind],
                boxmean=True,
            ))
        fig.update_layout(title="Distribution of actual indication splits (labeled hospitals)", yaxis_title="Share")
        st.plotly_chart(_theme(fig, 360), use_container_width=True, theme="streamlit")

    # Raw data table
    st.markdown('<p class="section-title">Raw data sample</p>', unsafe_allow_html=True)
    display_cols = [
        "total_6m_sales",
        "total_touchpoints_a", "total_touchpoints_b", "total_touchpoints_c",
        "total_hcps_a", "total_hcps_b", "total_hcps_c",
        "avg_split_a", "avg_split_b", "avg_split_c",
    ]
    available = [c for c in display_cols if c in modeling_df.columns]
    st.dataframe(
        modeling_df[available].head(50).round(4),
        use_container_width=True,
        hide_index=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    add_app_style()
    bundle = get_demo_bundle()
    page, active_model_key = render_sidebar(bundle)

    if page == "Overview":
        render_overview(bundle)
    elif page == "Model Comparison":
        render_model_comparison(bundle)
    elif page == "Calculator":
        render_calculator(bundle)
    elif page == "Data Explorer":
        render_data_explorer(bundle)


if __name__ == "__main__":
    main()