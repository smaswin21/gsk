from __future__ import annotations

import time
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gsk_model import (
    build_modeling_frame,
    fit_alr_models,
    get_feature_frame,
    get_target_frame,
    load_exported_artifacts,
    predict_alr,
    predict_multinomial,
    prepare_app_features_from_inputs,
    train_and_export,
)


PAGE_TITLE = "Sales by Indication"
MODEL_OPTIONS = {
    "Multinomial Logistic Regression": "multinomial",
    "ALR OLS Benchmark": "alr",
}
INDICATION_COLORS = {"A": "#1E257F", "B": "#FF6A00", "C": "#2AA198"}


st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def add_app_style() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(255, 153, 0, 0.08), transparent 24%),
                    var(--background-color);
            }
            [data-testid="stHeader"] {
                background: transparent;
            }
            [data-testid="stSidebar"] {
                border-right: 1px solid rgba(127, 127, 127, 0.14);
            }
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
                line-height: 1.45;
            }
            .sidebar-brand {
                margin: 0.35rem 0 1.2rem;
            }
            .sidebar-logo {
                font-size: 1.7rem;
                font-weight: 900;
                letter-spacing: 0.03em;
                color: #FF6A00;
                margin: 0;
                line-height: 1;
            }
            .sidebar-subtitle {
                margin: 0.3rem 0 0;
                color: var(--text-color);
                font-size: 1rem;
                line-height: 1.3;
            }
            .sidebar-section {
                font-size: 0.82rem;
                color: rgba(127, 127, 127, 0.95);
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin: 1.15rem 0 0.45rem;
                font-weight: 700;
            }
            .content-title {
                margin: 0;
                font-size: 2rem;
                line-height: 1.05;
                letter-spacing: -0.03em;
            }
            .content-subtitle {
                color: rgba(127, 127, 127, 0.95);
                margin-top: 0.35rem;
                margin-bottom: 1.25rem;
                font-size: 1rem;
                line-height: 1.45;
            }
            .note-card {
                background: var(--secondary-background-color);
                border: 1px solid rgba(127, 127, 127, 0.18);
                border-radius: 16px;
                padding: 0.95rem 1rem;
                margin: 0.35rem 0 1rem;
            }
            .note-card p {
                margin: 0;
                color: var(--text-color);
                line-height: 1.45;
            }
            .summary-card {
                background: var(--secondary-background-color);
                border: 1px solid rgba(127, 127, 127, 0.18);
                border-radius: 18px;
                padding: 1rem 1.05rem;
                height: 100%;
            }
            .summary-card h3 {
                margin: 0 0 0.4rem;
                color: #FF6A00;
                text-transform: uppercase;
                font-size: 0.82rem;
                letter-spacing: 0.08em;
            }
            .summary-card p {
                margin: 0;
                line-height: 1.45;
            }
            .metric-chip-wrap {
                display: flex;
                gap: 0.65rem;
                flex-wrap: wrap;
                margin-bottom: 1rem;
            }
            .metric-chip {
                background: rgba(255, 106, 0, 0.08);
                border: 1px solid rgba(255, 106, 0, 0.14);
                border-radius: 999px;
                padding: 0.55rem 0.85rem;
                color: var(--text-color);
                font-weight: 600;
                font-size: 0.94rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_demo_bundle() -> dict[str, Any]:
    train_and_export()
    multinomial_model, config = load_exported_artifacts()
    modeling_df = build_modeling_frame(config["data_path"])
    alr_models = fit_alr_models(get_feature_frame(modeling_df), get_target_frame(modeling_df))
    default_raw_inputs = {
        "total_6m_sales": int(round(modeling_df["total_6m_sales"].median())),
        "touchpoints_a": int(round(modeling_df["total_touchpoints_a"].median())),
        "touchpoints_b": int(round(modeling_df["total_touchpoints_b"].median())),
        "touchpoints_c": int(round(modeling_df["total_touchpoints_c"].median())),
        "hcps_a": int(round(modeling_df["total_hcps_a"].median())),
        "hcps_b": int(round(modeling_df["total_hcps_b"].median())),
        "hcps_c": int(round(modeling_df["total_hcps_c"].median())),
    }
    return {
        "multinomial_model": multinomial_model,
        "alr_models": alr_models,
        "config": config,
        "modeling_df": modeling_df,
        "sales_cv_default": float(config["feature_ranges"]["sales_cv"]["median"]),
        "default_raw_inputs": default_raw_inputs,
    }


def build_model_inputs_from_raw_inputs(raw_inputs: dict[str, float], sales_cv_default: float) -> pd.DataFrame:
    total_touchpoints_all = raw_inputs["touchpoints_a"] + raw_inputs["touchpoints_b"] + raw_inputs["touchpoints_c"]
    total_hcps_all = raw_inputs["hcps_a"] + raw_inputs["hcps_b"] + raw_inputs["hcps_c"]

    model_inputs = {
        "total_6m_sales": raw_inputs["total_6m_sales"],
        "sales_cv": sales_cv_default,
        "total_touchpoints_all": total_touchpoints_all,
        "touchpoints_share_a": raw_inputs["touchpoints_a"] / (total_touchpoints_all + 1e-6),
        "touchpoints_share_b": raw_inputs["touchpoints_b"] / (total_touchpoints_all + 1e-6),
        "total_hcps_all": total_hcps_all,
        "hcp_share_a": raw_inputs["hcps_a"] / (total_hcps_all + 1e-6),
        "hcp_share_b": raw_inputs["hcps_b"] / (total_hcps_all + 1e-6),
    }
    return prepare_app_features_from_inputs(model_inputs)


def predict_scenario(bundle: dict[str, Any], raw_inputs: dict[str, float], model_key: str) -> pd.Series:
    features = build_model_inputs_from_raw_inputs(raw_inputs, bundle["sales_cv_default"])
    if model_key == "alr":
        prediction_df = predict_alr(bundle["alr_models"], features)
    else:
        prediction_df = predict_multinomial(bundle["multinomial_model"], features)
    return prediction_df.iloc[0]


def apply_plot_theme(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=58, b=10),
        font=dict(size=14),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(zeroline=False)
    fig.update_yaxes(zeroline=False)
    return fig


def build_predicted_units_chart(prediction: pd.Series, total_sales: float) -> go.Figure:
    labels = ["A", "B", "C"]
    units = [
        prediction["pred_split_a"] * total_sales,
        prediction["pred_split_b"] * total_sales,
        prediction["pred_split_c"] * total_sales,
    ]
    max_units = max(units) if units else 0
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=units,
            marker_color=[INDICATION_COLORS[label] for label in labels],
            text=[f"{value:.0f} units" for value in units],
            textposition="outside",
            hovertemplate="Indication %{x}<br>%{y:.0f} units<extra></extra>",
        )
    )
    fig.update_layout(
        title="Predicted units by indication",
        showlegend=False,
        yaxis=dict(title="Predicted units", range=[0, max_units * 1.18 if max_units > 0 else 1]),
        xaxis=dict(title="Indication"),
    )
    return apply_plot_theme(fig, 350)


def build_predicted_mix_chart(prediction: pd.Series) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Indication A", "Indication B", "Indication C"],
                values=[
                    prediction["pred_split_a"] * 100,
                    prediction["pred_split_b"] * 100,
                    prediction["pred_split_c"] * 100,
                ],
                hole=0.56,
                marker=dict(colors=[INDICATION_COLORS["A"], INDICATION_COLORS["B"], INDICATION_COLORS["C"]]),
                textinfo="label+percent",
                hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>",
                sort=False,
            )
        ]
    )
    fig.update_layout(
        title="Predicted mix by indication",
        showlegend=False,
    )
    return apply_plot_theme(fig, 330)


def initialize_sidebar_state(defaults: dict[str, int]) -> None:
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "submitted_raw_inputs" not in st.session_state:
        st.session_state["submitted_raw_inputs"] = {key: float(value) for key, value in defaults.items()}
    if "submitted_model_key" not in st.session_state:
        st.session_state["submitted_model_key"] = "multinomial"


def render_sidebar(bundle: dict[str, Any]) -> tuple[str, str]:
    initialize_sidebar_state(bundle["default_raw_inputs"])

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <p class="sidebar-logo">GSK</p>
                <p class="sidebar-subtitle">Hospital Sales Allocation</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
        page = st.radio(
            "Navigation",
            ["Executive Summary", "Calculator"],
            label_visibility="collapsed",
        )

        st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)
        model_label = st.selectbox(
            "Model",
            list(MODEL_OPTIONS.keys()),
            label_visibility="collapsed",
        )

        st.markdown('<div class="sidebar-section">Filters</div>', unsafe_allow_html=True)
        with st.form("sidebar_calculator_form"):
            total_6m_sales = st.number_input(
                "Total sales (units)",
                min_value=1,
                step=25,
                value=int(st.session_state["total_6m_sales"]),
            )

            st.markdown("**Touchpoints by indication**")
            touchpoints_a = st.number_input("Touchpoints A", min_value=0, step=5, value=int(st.session_state["touchpoints_a"]))
            touchpoints_b = st.number_input("Touchpoints B", min_value=0, step=5, value=int(st.session_state["touchpoints_b"]))
            touchpoints_c = st.number_input("Touchpoints C", min_value=0, step=5, value=int(st.session_state["touchpoints_c"]))

            st.markdown("**HCPs reached by indication**")
            hcps_a = st.number_input("HCPs A", min_value=0, step=5, value=int(st.session_state["hcps_a"]))
            hcps_b = st.number_input("HCPs B", min_value=0, step=5, value=int(st.session_state["hcps_b"]))
            hcps_c = st.number_input("HCPs C", min_value=0, step=5, value=int(st.session_state["hcps_c"]))

            submitted = st.form_submit_button("Run Model", use_container_width=True, type="primary")

        if submitted:
            latest_inputs = {
                "total_6m_sales": float(total_6m_sales),
                "touchpoints_a": float(touchpoints_a),
                "touchpoints_b": float(touchpoints_b),
                "touchpoints_c": float(touchpoints_c),
                "hcps_a": float(hcps_a),
                "hcps_b": float(hcps_b),
                "hcps_c": float(hcps_c),
            }
            st.session_state["submitted_raw_inputs"] = latest_inputs
            st.session_state["submitted_model_key"] = MODEL_OPTIONS[model_label]
            for key, value in latest_inputs.items():
                st.session_state[key] = value
            with st.spinner(f"Calculating predictions using {model_label}..."):
                time.sleep(0.35)

        return page, MODEL_OPTIONS[model_label]


def render_summary_card(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="summary-card">
            <h3>{title}</h3>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_executive_summary(bundle: dict[str, Any], model_label: str) -> None:
    st.markdown('<h1 class="content-title">Executive Summary</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="content-subtitle">Estimate how hospital sales split across Indications A, B, and C using commercial activity and HCP reach at hospital level.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="metric-chip-wrap">
            <div class="metric-chip">Validated on held-out hospitals</div>
            <div class="metric-chip">Predicted shares always sum to 100%</div>
            <div class="metric-chip">Selected model updates the prediction view</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    top = st.columns(2)
    with top[0]:
        render_summary_card(
            "Context",
            "Hospitals buy one drug that may be used for multiple indications, but the order does not specify how much was used for A, B, or C.",
        )
    with top[1]:
        render_summary_card(
            "Goal",
            "Break total hospital sales into an estimated indication split so commercial effectiveness can be discussed by disease area.",
        )
    bottom = st.columns(2)
    with bottom[0]:
        render_summary_card(
            "Available Data",
            "Total sales, touchpoints by indication, HCP reach by indication, and known indication split labels for a training sample.",
        )
    with bottom[1]:
        render_summary_card(
            "Current Prediction View",
            f"The calculator is currently configured to score scenarios with {model_label}. Use the sidebar to switch models and recalculate.",
        )


def render_calculator_view(bundle: dict[str, Any], active_model_key: str) -> None:
    submitted_raw_inputs = st.session_state["submitted_raw_inputs"]
    model_key_for_prediction = st.session_state.get("submitted_model_key", active_model_key)
    model_label = next(label for label, key in MODEL_OPTIONS.items() if key == model_key_for_prediction)
    prediction = predict_scenario(bundle, submitted_raw_inputs, model_key_for_prediction)

    st.markdown('<h1 class="content-title">Predictions</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="content-subtitle">Use the sidebar calculator inputs and model dropdown to update the hospital-level prediction view.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="note-card">
            <p><strong>Model used:</strong> {model_label}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        build_predicted_units_chart(prediction, submitted_raw_inputs["total_6m_sales"]),
        use_container_width=True,
        theme="streamlit",
    )
    st.markdown(
        """
        <div class="note-card">
            <p>This chart shows the predicted number of units allocated to Indications A, B, and C for the hospital scenario submitted from the sidebar filters.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        build_predicted_mix_chart(prediction),
        use_container_width=True,
        theme="streamlit",
    )


def main() -> None:
    add_app_style()
    bundle = get_demo_bundle()
    page, active_model_key = render_sidebar(bundle)

    if page == "Executive Summary":
        model_label = next(label for label, key in MODEL_OPTIONS.items() if key == active_model_key)
        render_executive_summary(bundle, model_label)
    else:
        render_calculator_view(bundle, active_model_key)


if __name__ == "__main__":
    main()
