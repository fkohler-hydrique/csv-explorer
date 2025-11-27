# app.py
import io
from typing import List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.api.types import is_object_dtype


# ------------- Utility functions ------------- #

def load_csv(file_or_path, sep: Optional[str] = None, decimal: str = ".") -> pd.DataFrame:
    """Load a delimited text file from an uploaded file or a path string.

    By default, the separator is auto-detected (comma / semicolon / tab / pipe),
    but an explicit separator and decimal character can be passed in.
    """
    if file_or_path is None:
        raise ValueError("No file or path provided.")

    # Build read_csv keyword arguments
    read_kwargs = {"decimal": decimal}

    if sep is None:
        # Auto-detect separator; requires the Python engine
        read_kwargs.update(sep=None, engine="python")
    else:
        # Use the user-specified separator
        read_kwargs["sep"] = sep

    if hasattr(file_or_path, "read"):
        # Uploaded file via Streamlit
        return pd.read_csv(file_or_path, **read_kwargs)
    else:
        # Path string
        return pd.read_csv(file_or_path, **read_kwargs)


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

    # If the configured date column is not valid, ignore it.
    if date_col not in df.columns:
        date_col = None

    # If no configured date column, try to guess one.
    if date_col is None:
        for candidate in df.columns:
            if "date" in candidate.lower():
                try:
                    pd.to_datetime(df[candidate])
                    date_col = candidate
                    break
                except (ValueError, TypeError):
                    continue

    # Final validation
    if date_col is not None:
        try:
            pd.to_datetime(df[date_col])
        except (ValueError, TypeError):
            date_col = None

    return date_col


def parse_dates(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    """Convert date_col to datetime if present."""
    if date_col and date_col in df.columns:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df


def filter_by_date(
    df: pd.DataFrame,
    date_col: Optional[str],
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    year: Optional[int],
    month: Optional[int],
) -> pd.DataFrame:
    """Apply date filters if a valid date column exists."""
    if date_col is None or date_col not in df.columns:
        return df

    filtered = df.copy()
    if start_date is not None:
        filtered = filtered[filtered[date_col] >= start_date]
    if end_date is not None:
        filtered = filtered[filtered[date_col] <= end_date]
    if year is not None:
        filtered = filtered[filtered[date_col].dt.year == year]
    if month is not None:
        filtered = filtered[filtered[date_col].dt.month == month]

    return filtered


def get_date_range(df: pd.DataFrame, date_col: Optional[str]):
    """Return (min_date, max_date) if date_col is valid, otherwise (None, None)."""
    if date_col is None or date_col not in df.columns:
        return None, None
    try:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        return dates.min(), dates.max()
    except (ValueError, TypeError):
        return None, None


def plot_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_cols: List[str],
    template: str = "plotly_white",
):
    """Create a time-series line chart with Plotly."""
    melted = df.melt(id_vars=date_col, value_vars=value_cols, var_name="variable", value_name="value")
    fig = px.line(
        melted,
        x=date_col,
        y="value",
        color="variable",
        template=template,
    )
    fig.update_layout(
        title="Time Series",
        xaxis_title=date_col,
        yaxis_title="Value",
        legend_title="Series",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)
    return fig


def make_figure(
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    plot_type: str,
    template: str = "plotly_white",
):
    """Create a Plotly figure based on the selected plot_type."""
    if not y_cols:
        raise ValueError("No y columns selected.")

    if len(y_cols) == 1:
        y = y_cols[0]

        if plot_type == "line":
            fig = px.line(df, x=x_col, y=y, template=template)
        elif plot_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y, template=template)
        elif plot_type == "bar":
            fig = px.bar(df, x=x_col, y=y, template=template)
        elif plot_type == "histogram":
            fig = px.histogram(df, x=y, template=template)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
    else:
        melted = df.melt(
            id_vars=x_col,
            value_vars=y_cols,
            var_name="series",
            value_name="value",
        )

        if plot_type == "line":
            fig = px.line(
                melted,
                x=x_col,
                y="value",
                color="series",
                template=template,
            )
        elif plot_type == "scatter":
            fig = px.scatter(
                melted,
                x=x_col,
                y="value",
                color="series",
                template=template,
            )
        elif plot_type == "bar":
            fig = px.bar(
                melted,
                x=x_col,
                y="value",
                color="series",
                barmode="group",
                template=template,
            )
        else:
            raise ValueError(
                "For multiple Y columns, only line, scatter, or bar are supported."
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


def save_plot_config(
    saved_plots: List[dict],
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    plot_type: str,
    template: str,
):
    """Append a new plot configuration to the list of saved plots."""
    config = {
        "data": df.copy(),
        "x_col": x_col,
        "y_cols": y_cols,
        "plot_type": plot_type,
        "template": template,
    }
    saved_plots.append(config)


# ------------- Main app ------------- #

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

    # --- File reading options (sidebar) --- #
    st.sidebar.subheader("File reading options")

    sep_label = st.sidebar.selectbox(
        "Column separator",
        ["Auto-detect", "Comma (,)", "Semicolon (;)", "Tab (\\t)", "Pipe (|)"],
        index=0,
    )

    sep_map = {
        "Auto-detect": None,
        "Comma (,)": ",",
        "Semicolon (;)": ";",
        "Tab (\\t)": "\t",
        "Pipe (|)": "|",
    }
    sep = sep_map[sep_label]

    decimal_label = st.sidebar.selectbox(
        "Decimal separator",
        ["Dot (.)", "Comma (,)"],
        index=0,
    )
    decimal = "." if decimal_label == "Dot (.)" else ","

    if not uploaded_file and not csv_path:
        st.info("Please upload a CSV or enter a valid path to begin.")
        st.stop()

    try:
        df = load_csv(uploaded_file or csv_path, sep=sep, decimal=decimal)
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

        st.write("**Summary statistics (numeric columns):**")
        st.write(df.describe(include="number").T)

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
    all_cols = df.columns.tolist()

    col_x, col_y, col_type = st.columns(3)

    with col_x:
        x_col = st.selectbox("X-axis column", options=all_cols)

    with col_y:
        y_cols = st.multiselect(
            "Y-axis column(s)",
            options=numeric_cols,
            default=numeric_cols[:1],
            help="Only numeric columns can be selected for Y.",
        )

    with col_type:
        plot_type = st.selectbox("Plot type", ["line", "scatter", "bar", "histogram"])

    template = st.selectbox(
        "Plotly theme",
        ["plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"],
        index=0,
    )

    # --- Date filters (if date column is present) --- #
    df = parse_dates(df, date_col)
    min_date, max_date = get_date_range(df, date_col)

    start_date = None
    end_date = None
    year = None
    month = None

    if date_col and min_date is not None and max_date is not None:
        st.sidebar.subheader("Date filtering")

        use_date_range = st.sidebar.checkbox(
            "Filter by date range", value=False
        )

        if use_date_range:
            start_date = st.sidebar.date_input(
                "Start date", value=min_date.date()
            )
            end_date = st.sidebar.date_input(
                "End date", value=max_date.date()
            )
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

        use_year_month = st.sidebar.checkbox(
            "Filter by year/month", value=False
        )

        if use_year_month:
            years = sorted(df[date_col].dt.year.dropna().unique())
            year = st.sidebar.selectbox("Year", options=[None] + list(years), index=0)

            if year is not None:
                months = sorted(
                    df.loc[df[date_col].dt.year == year, date_col].dt.month.dropna().unique()
                )
                month = st.sidebar.selectbox(
                    "Month", options=[None] + list(months), index=0
                )

    filtered_df = filter_by_date(df, date_col, start_date, end_date, year, month)

    st.write(f"**Filtered data shape:** {filtered_df.shape[0]} rows × {filtered_df.shape[1]} columns")

    # --- Plotting --- #
    st.subheader("📈 Visualization")

    if not y_cols:
        st.warning("Please select at least one numeric Y column.")
    else:
        try:
            if date_col and x_col == date_col and plot_type == "line":
                fig = plot_time_series(filtered_df, date_col, y_cols, template=template)
            else:
                fig = make_figure(filtered_df, x_col, y_cols, plot_type, template=template)

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating plot: {e}")

    # --- Save plots --- #
    st.subheader("💾 Save plot configurations")

    if "saved_plots" not in st.session_state:
        st.session_state["saved_plots"] = []

    if st.button("Save current plot configuration"):
        try:
            save_plot_config(
                st.session_state["saved_plots"],
                filtered_df,
                x_col,
                y_cols,
                plot_type,
                template,
            )
            st.success("Plot configuration saved!")
        except Exception as e:
            st.error(f"Could not save plot configuration: {e}")

    if st.session_state["saved_plots"]:
        st.subheader("🗂 Saved plots")
        for i, p in enumerate(st.session_state["saved_plots"], start=1):
            st.markdown(f"**Saved plot #{i}:** {', '.join(p['y_cols'])} vs {p['x_col']} ({p['plot_type']})")
            fig_saved = make_figure(
                p["data"],
                p["x_col"],
                p["y_cols"],
                p["plot_type"],
                template=p["template"],
            )
            st.plotly_chart(fig_saved, use_container_width=True, key=f"saved_plot_{i}")

    # --- Keep / discard plots --- #
    st.subheader("🧹 Manage plots")

    if st.session_state["saved_plots"]:
        keep_indices = []
        for i, p in enumerate(st.session_state["saved_plots"], start=1):
            keep = st.checkbox(
                f"Keep plot #{i}: {', '.join(p['y_cols'])} vs {p['x_col']} ({p['plot_type']})",
                value=True,
            )
            if keep:
                keep_indices.append(i - 1)

        if st.button("Apply keep/discard selection"):
            st.session_state["saved_plots"] = [
                p for j, p in enumerate(st.session_state["saved_plots"]) if j in keep_indices
            ]
            st.success("Updated saved plots list!")

    if st.session_state["saved_plots"]:
        st.subheader("📚 Kept plots")
        for i, p in enumerate(st.session_state["saved_plots"], start=1):
            st.markdown(f"**Kept plot #{i}:** {', '.join(p['y_cols'])} vs {p['x_col']} ({p['plot_type']})")
            fig_saved = make_figure(
                p["data"],
                p["x_col"],
                p["y_cols"],
                p["plot_type"],
                template=p["template"],
            )
            st.plotly_chart(fig_saved, use_container_width=True, key=f"kept_plot_{i}")


if __name__ == "__main__":
    main()
