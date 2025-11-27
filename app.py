# app.py
import io
from typing import List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.api.types import is_object_dtype


# ------------- Utility functions ------------- #

def load_csv(file_or_path) -> pd.DataFrame:
    """Load a CSV from an uploaded file or a path string."""
    if file_or_path is None:
        raise ValueError("No file or path provided.")

    if hasattr(file_or_path, "read"):
        # Uploaded file via Streamlit
        return pd.read_csv(file_or_path)
    else:
        # Path string
        return pd.read_csv(file_or_path)

def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object columns to plain Python strings so Streamlit / Arrow
    don't crash or spam warnings when serializing the dataframe.
    """
    safe = df.copy()
    for col in safe.columns:
        if is_object_dtype(safe[col].dtype):
            safe[col] = safe[col].astype(str)  # classic Python string dtype
    return safe


def detect_date_column(df: pd.DataFrame, configured: Optional[str] = None) -> Optional[str]:
    """Try to detect a date column; prefer configured name if present."""
    date_col = configured

    if date_col is not None and date_col not in df.columns:
        date_col = None

    if date_col is None:
        for col in df.columns:
            if col.lower() == "date":
                date_col = col
                break

    if date_col is None:
        return None

    # Try to parse
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    if parsed.isna().all():
        return None

    df[date_col] = parsed
    return date_col


def filter_by_year_month(
    df: pd.DataFrame, date_col: str, year: Optional[int], month: Optional[int]
) -> pd.DataFrame:
    """Filter dataframe by year and/or month if selected."""
    if date_col is None:
        return df

    filtered = df.copy()
    if year is not None:
        filtered = filtered[filtered[date_col].dt.year == year]
    if month is not None:
        filtered = filtered[filtered[date_col].dt.month == month]

    return filtered

def apply_date_range(df: pd.DataFrame, date_col: str, date_range) -> pd.DataFrame:
    """Filter dataframe by a start/end date from a Streamlit date_input."""
    if date_col is None or not date_range:
        return df

    start, end = date_range
    if start is None or end is None:
        return df

    mask = (df[date_col] >= pd.Timestamp(start)) & (df[date_col] <= pd.Timestamp(end))
    return df[mask]


def apply_resample(df: pd.DataFrame, date_col: str, y_cols: List[str], rule: str) -> pd.DataFrame:
    """Resample time series by rule (e.g. 'D', 'W', 'M')."""
    if date_col is None or rule == "None":
        return df

    df_resampled = (
        df.set_index(date_col)[y_cols]
        .resample(rule)
        .mean()
        .reset_index()
    )
    # Keep any extra non-numeric columns? For simplicity, we don't.
    return df_resampled


def apply_rolling(df: pd.DataFrame, y_cols: List[str], window: int) -> pd.DataFrame:
    """Apply rolling mean to Y columns."""
    if window <= 1:
        return df

    df_rolled = df.copy()
    for col in y_cols:
        df_rolled[col] = df_rolled[col].rolling(window=window, min_periods=1).mean()
    return df_rolled


def make_figure(
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    plot_type: str,
    color_col: Optional[str] = None,
    template: str = "plotly_white",
):
    """Build a Plotly figure for given axes and type."""
    if not y_cols:
        return None

    # For multi-Y, we melt the dataframe
    plot_df = df[[x_col] + y_cols].copy()
    melted = plot_df.melt(id_vars=x_col, value_vars=y_cols, var_name="series", value_name="value")

    if plot_type == "Line":
        fig = px.line(
            melted,
            x=x_col,
            y="value",
            color="series",
            template=template,
        )
    elif plot_type == "Scatter":
        fig = px.scatter(
            melted,
            x=x_col,
            y="value",
            color="series",
            template=template,
        )
    elif plot_type == "Bar":
        fig = px.bar(
            melted,
            x=x_col,
            y="value",
            color="series",
            template=template,
            barmode="group",
        )
    else:
        fig = px.line(
            melted,
            x=x_col,
            y="value",
            color="series",
            template=template,
        )

    fig.update_layout(
        title=f"{', '.join(y_cols)} vs {x_col}",
        xaxis_title=x_col,
        yaxis_title=" / ".join(y_cols),
        legend_title="Series",
        margin=dict(l=40, r=20, t=40, b=40),
    )

    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)

    return fig


# ------------- Streamlit app ------------- #

def main():
    st.set_page_config(
        page_title="Interactive CSV Explorer",
        layout="wide",
    )

    st.title("📊 Interactive CSV Explorer")
    st.markdown(
        "Upload a CSV or provide a file path, then interactively explore and visualize your data."
    )

    # --- File input section --- #
    col_file, col_path = st.columns(2)

    with col_file:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    with col_path:
        csv_path = st.text_input("...or enter a CSV file path")

    if not uploaded_file and not csv_path:
        st.info("Please upload a CSV or enter a valid path to begin.")
        st.stop()

    try:
        df = load_csv(uploaded_file or csv_path)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.success(f"Loaded data with shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # --- Data overview --- #
    with st.expander("🔍 Data preview & summary", expanded=True):
        st.write("**Head:**")
        st.dataframe(make_arrow_safe(df.head()))

        st.write("**Column types:**")
        st.write(df.dtypes.astype(str))

        if st.toggle("Show basic statistics"):
            stats = df.describe(include="all")
            st.dataframe(make_arrow_safe(stats))

    # --- Detect date column --- #
    st.sidebar.header("Settings")

    configured_date_col = st.sidebar.text_input("Specify date column (optional)", value="")
    configured_date_col = configured_date_col or None

    date_col = detect_date_column(df, configured_date_col)

    if date_col:
        st.sidebar.success(f"Using '{date_col}' as date column.")
    else:
        st.sidebar.info("No valid date column detected. Date filters disabled.")

    # --- Axis & plot controls --- #
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found to plot.")
        st.stop()

    # X-axis options: prefer date column
    x_axis_options = []
    if date_col is not None:
        x_axis_options.append(date_col)
    x_axis_options.extend([c for c in df.columns if c != date_col])

    st.sidebar.subheader("Axes & Plot")

    x_col = st.sidebar.selectbox("X axis", options=x_axis_options)
    y_cols = st.sidebar.multiselect(
        "Y axis (numeric)", options=numeric_cols, default=numeric_cols[:1]
    )

    plot_type = st.sidebar.selectbox("Plot type", ["Line", "Scatter", "Bar"])

    template = st.sidebar.selectbox("Plot theme", ["plotly_white", "plotly_dark", "ggplot2"])

    # --- Time controls --- #
    if date_col is not None:
        st.sidebar.subheader("Time filtering")

        years = sorted(df[date_col].dropna().dt.year.unique())
        year = st.sidebar.selectbox("Year (optional)", options=["All"] + years, index=0)
        if year == "All":
            year = None

        months = list(range(1, 13))
        month_labels = ["All", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_choice = st.sidebar.selectbox("Month (optional)", options=month_labels, index=0)

        if month_choice == "All":
            month = None
        else:
            month = month_labels.index(month_choice)  # 1-based

        # Optional date-range filter
        if st.sidebar.checkbox("Use date range filter instead of year/month"):
            min_date = df[date_col].min()
            max_date = df[date_col].max()
            date_range = st.sidebar.date_input(
                "Date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
        else:
            date_range = None
    else:
        year = None
        month = None
        date_range = None

    # --- Resampling & smoothing --- #
    st.sidebar.subheader("Aggregation & smoothing")

    resample_rule = st.sidebar.selectbox(
        "Resample (if X is date)",
        options=["None", "Day", "Week", "Month"],
        help="Aggregate numeric Y columns over time.",
    )
    rule_map = {"None": "None", "Day": "D", "Week": "W", "Month": "M"}

    rolling_window = st.sidebar.slider(
        "Rolling mean window (points)", min_value=1, max_value=60, value=1
    )

    # --- Filtering and transformation --- #
    df_filtered = df.copy()

    if date_col is not None:
        df_filtered = filter_by_year_month(df_filtered, date_col, year, month)
        df_filtered = apply_date_range(df_filtered, date_col, date_range)

    # Resample only if x_col is the date column and a rule is chosen
    if date_col is not None and x_col == date_col and resample_rule != "None":
        df_filtered = apply_resample(df_filtered, date_col, y_cols, rule_map[resample_rule])

    if rolling_window > 1 and y_cols:
        df_filtered = apply_rolling(df_filtered, y_cols, window=rolling_window)

    # --- Main plot area --- #
    left, right = st.columns([3, 1])

    with left:
        st.subheader("Live plot")
        if not y_cols:
            st.info("Select at least one Y-axis column to see a plot.")
        else:
            fig = make_figure(df_filtered, x_col, y_cols, plot_type, template=template)
            if fig is not None:
                st.plotly_chart(fig, width="stretch", key="live_plot")
            else:
                st.info("Could not build figure – check your selections.")

    with right:
        st.subheader("Actions")

        # Initialize saved plots in session state
        if "saved_plots" not in st.session_state:
            st.session_state["saved_plots"] = []

        if st.button("📌 Keep current plot"):
            if y_cols:
                st.session_state["saved_plots"].append(
                    {
                        "x_col": x_col,
                        "y_cols": list(y_cols),
                        "plot_type": plot_type,
                        "template": template,
                        "data": df_filtered.copy(),
                    }
                )
                st.success("Plot kept.")
            else:
                st.warning("Select at least one Y column before keeping a plot.")

        # Download filtered data
        if st.button("⬇️ Download filtered data as CSV"):
            buffer = io.StringIO()
            df_filtered.to_csv(buffer, index=False)
            st.download_button(
                "Download now",
                data=buffer.getvalue(),
                file_name="filtered_data.csv",
                mime="text/csv",
            )

    # --- Display kept plots --- #
    if st.session_state["saved_plots"]:
        st.subheader("📌 Kept plots")
        for i, p in enumerate(st.session_state["saved_plots"], start=1):
            st.markdown(f"**Kept plot #{i}:** {', '.join(p['y_cols'])} vs {p['x_col']} ({p['plot_type']})")
            fig_saved = make_figure(
                p["data"],
                p["x_col"],
                p["y_cols"],
                p["plot_type"],
                template=p["template"],
            )
            st.plotly_chart(fig_saved, width="stretch", key=f"kept_plot_{i}")



if __name__ == "__main__":
    main()
