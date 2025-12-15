# import io
import os
import pickle
from typing import List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.api.types import is_object_dtype
from plotly.subplots import make_subplots

# ------------- Configuration & constants ------------- #

CANDIDATE_SEPARATORS = [",", ";", "\t", "|"]
DATE_KEYWORDS = ("date", "time", "datum", "zeit", "datetime")
SAVED_PLOTS_FILE = "saved_plots.pkl"


def detect_separator_from_sample(sample: str, decimal: str = ".") -> Optional[str]:
    """Heuristically detect the field separator from a text sample."""
    lines = [line for line in sample.splitlines() if line.strip()]
    if not lines:
        return None

    best_sep: Optional[str] = None
    best_score: float = 0.0

    for sep in CANDIDATE_SEPARATORS:
        if sep == decimal:
            continue

        counts = [line.count(sep) for line in lines[:25]]
        if not counts or max(counts) == 0:
            continue

        mean_count = sum(counts) / len(counts)
        variance = sum((c - mean_count) ** 2 for c in counts) / len(counts)
        score = mean_count - variance
        if score > best_score:
            best_score = score
            best_sep = sep

    return best_sep


def parse_dates_flexible(
    df: pd.DataFrame,
    date_col: Optional[str],
    dayfirst: bool = False,
    date_format: Optional[str] = None,
) -> pd.DataFrame:
    """Parse dates with optional manual overrides."""
    if date_col is None or date_col not in df.columns:
        return df

    df = df.copy()

    try:
        if date_format:
            df[date_col] = pd.to_datetime(
                df[date_col],
                format=date_format,
                errors="coerce",
            )
        else:
            df[date_col] = pd.to_datetime(
                df[date_col],
                dayfirst=dayfirst,
                # infer_datetime_format=True,
                errors="coerce",
            )
    except Exception:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    return df


def load_saved_plots_from_disk() -> List[dict]:
    """Load saved plots from disk if available."""
    if not os.path.exists(SAVED_PLOTS_FILE):
        return []
    try:
        with open(SAVED_PLOTS_FILE, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, list):
            return [p for p in data if isinstance(p, dict)]
    except Exception:
        pass
    return []


def persist_saved_plots_to_disk(saved_plots: List[dict]) -> None:
    """Persist the list of saved plots to disk (best-effort)."""
    try:
        with open(SAVED_PLOTS_FILE, "wb") as f:
            pickle.dump(saved_plots, f)
    except Exception:
        pass


# ------------- Utility functions ------------- #

def load_csv(
    file_or_path,
    sep: Optional[str] = None,
    decimal: str = ".",
    header="infer",
    skiprows=None,
) -> pd.DataFrame:
    """Load a delimited text file from an uploaded file or a path string.

    Tries UTF-8 first, then falls back to latin-1 if decoding fails.
    """
    if file_or_path is None:
        raise ValueError("No file or path provided.")

    read_kwargs = {"decimal": decimal, "header": header}
    if skiprows is not None:
        read_kwargs["skiprows"] = skiprows

    if sep is None:
        read_kwargs.update(sep=None, engine="python")
    else:
        read_kwargs["sep"] = sep

    def _read_with_encoding(enc: Optional[str]):
        # Copy kwargs so we don't mutate between attempts
        kwargs = dict(read_kwargs)
        if enc is not None:
            kwargs["encoding"] = enc
        # If it's an uploaded file, rewind the buffer before each attempt
        if hasattr(file_or_path, "read"):
            try:
                file_or_path.seek(0)
            except Exception:
                pass
            return pd.read_csv(file_or_path, **kwargs)
        else:
            return pd.read_csv(file_or_path, **kwargs)

    # Try UTF-8 first, then latin-1 as a fallback
    try:
        return _read_with_encoding("utf-8")
    except UnicodeDecodeError:
        return _read_with_encoding("latin-1")


def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns to plain Python strings for Streamlit/Arrow."""
    safe = df.copy()
    for col in safe.columns:
        if is_object_dtype(safe[col].dtype):
            safe[col] = safe[col].astype(str)
    return safe


def detect_date_column(df: pd.DataFrame, configured: Optional[str] = None) -> Optional[str]:
    """Try to detect a date/time column; prefer configured name if present."""
    date_col = configured if configured in df.columns else None

    if date_col is None:
        for candidate in df.columns:
            lower = candidate.lower()
            if any(keyword in lower for keyword in DATE_KEYWORDS):
                try:
                    pd.to_datetime(df[candidate])
                    date_col = candidate
                    break
                except (ValueError, TypeError):
                    continue

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
    """Filter df by date range and/or year/month selections."""
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


def apply_secondary_yaxis(fig, styles: Optional[dict], enable_secondary: bool = False) -> None:
    """Assign traces to y or y2 based on styles[*]['axis'] (1 or 2), robustly.

    Handles cases where Plotly Express trace.name doesn't match the original column name,
    especially for single-series plots.
    """
    if not enable_secondary or not styles:
        return

    any_on_y2 = False

    # Helper: decide which style key applies to a trace
    def _style_key_for_trace(trace):
        name = getattr(trace, "name", None)

        # Direct match (most multi-series cases)
        if name and name in styles:
            return name

        # Try legendgroup as fallback
        lg = getattr(trace, "legendgroup", None)
        if lg and lg in styles:
            return lg

        # If there is only one style entry and one trace, assume they correspond
        if len(styles) == 1 and len(fig.data) == 1:
            return next(iter(styles.keys()))

        return None

    for trace in fig.data:
        key = _style_key_for_trace(trace)
        axis = 1
        if key is not None:
            axis = (styles.get(key, {}) or {}).get("axis", 1)

        if axis == 2:
            trace.update(yaxis="y2")
            any_on_y2 = True
        else:
            trace.update(yaxis="y")

    if any_on_y2:
        # Make sure the right axis is actually visible and has room
        current_margin = fig.layout.margin.to_plotly_json() if fig.layout.margin else {}
        r = current_margin.get("r", 20)
        fig.update_layout(
            margin=dict(
                l=current_margin.get("l", 40),
                r=max(r, 70),   # <-- important: room for right ticks/labels
                t=current_margin.get("t", 40),
                b=current_margin.get("b", 40),
            ),
            yaxis2=dict(
                overlaying="y",
                side="right",
                autorange=True,
                showgrid=False,
                zeroline=False,
                showline=True,
                ticks="outside",
                showticklabels=True,
                title=dict(text=""),  # keep empty unless you later add a UI label
            ),
        )


def apply_series_styles(fig, styles: Optional[dict]) -> None:
    """Apply per-series style (color, width, dash) to a Plotly figure."""
    if not styles:
        return

    def _style_key_for_trace(trace):
        name = getattr(trace, "name", None)
        if name and name in styles:
            return name

        lg = getattr(trace, "legendgroup", None)
        if lg and lg in styles:
            return lg

        if len(styles) == 1 and len(fig.data) == 1:
            return next(iter(styles.keys()))

        return None

    for trace in fig.data:
        key = _style_key_for_trace(trace)
        if key is None:
            continue

        s = styles.get(key) or {}
        line_kwargs = {}
        if s.get("color"):
            line_kwargs["color"] = s["color"]
        if s.get("width") is not None:
            line_kwargs["width"] = s["width"]
        if s.get("dash"):
            line_kwargs["dash"] = s["dash"]

        if line_kwargs:
            trace.update(line=line_kwargs)


def plot_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_cols: List[str],
    template: str = "plotly_white",
    title: Optional[str] = None,
    styles: Optional[dict] = None,
    enable_secondary_axis: bool = False,
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
    default_title = "Time Series"
    fig.update_layout(
        title=title or default_title,
        xaxis_title=date_col,
        yaxis_title="Value",
        legend_title="Series",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)

    apply_series_styles(fig, styles)
    apply_secondary_yaxis(fig, styles, enable_secondary=enable_secondary_axis)
    return fig


def make_figure(
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    plot_type: str,
    template: str = "plotly_white",
    title: Optional[str] = None,
    styles: Optional[dict] = None,
    enable_secondary_axis: bool = False,
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

    default_title = f"{', '.join(y_cols)} vs {x_col}"
    fig.update_layout(
        title=title or default_title,
        xaxis_title=x_col,
        yaxis_title=" / ".join(y_cols),
        legend_title="Series",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)

    apply_series_styles(fig, styles)
    apply_secondary_yaxis(fig, styles, enable_secondary=enable_secondary_axis)
    return fig


def save_plot_config(
    saved_plots: List[dict],
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    plot_type: str,
    template: str,
    label: Optional[str] = None,
    styles: Optional[dict] = None,
):
    """Append a new plot configuration to the list of saved plots."""
    styles_for_plot = None
    if styles:
        styles_for_plot = {k: v for k, v in styles.items() if k in y_cols}

    config = {
        "data": make_arrow_safe(df.copy()),
        "x_col": x_col,
        "y_cols": y_cols,
        "plot_type": plot_type,
        "template": template,
        "label": label,
        "styles": styles_for_plot,
    }
    saved_plots.append(config)
    persist_saved_plots_to_disk(saved_plots)


def compute_global_x_range(saved_plots: List[dict], indices: List[int]):
    """Compute global x-range (min, max) across given saved plots."""
    global_xmin = None
    global_xmax = None

    for idx in indices:
        p = saved_plots[idx]
        x_name = p["x_col"]
        if x_name not in p["data"].columns:
            continue
        s = p["data"][x_name]
        try:
            v = pd.to_datetime(s, errors="coerce")
            if v.notna().any():
                vals = v
            else:
                vals = pd.to_numeric(s, errors="coerce")
        except Exception:
            vals = pd.to_numeric(s, errors="coerce")
        vals = vals.dropna()
        if vals.empty:
            continue
        local_min = vals.min()
        local_max = vals.max()
        if global_xmin is None or local_min < global_xmin:
            global_xmin = local_min
        if global_xmax is None or local_max > global_xmax:
            global_xmax = local_max

    return global_xmin, global_xmax


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

    # --- Initialize session state --- #
    if "saved_plots" not in st.session_state:
        st.session_state["saved_plots"] = load_saved_plots_from_disk()
    if "series_styles" not in st.session_state:
        st.session_state["series_styles"] = {}

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

    # --- Header options (sidebar) --- #
    st.sidebar.subheader("Header options")

    header_label = st.sidebar.selectbox(
        "Header row(s)",
        [
            "First row only",
            "Second row only (skip first row)",
            "First two rows (multi-line header)",
            "No header (use generic column names)",
        ],
        index=0,
    )

    header = "infer"
    skiprows = None
    if header_label == "First row only":
        header = 0
        skiprows = None
    elif header_label == "Second row only (skip first row)":
        header = 0
        skiprows = [0]
    elif header_label == "First two rows (multi-line header)":
        header = [0, 1]
        skiprows = None
    elif header_label == "No header (use generic column names)":
        header = None
        skiprows = None

    if not uploaded_file and not csv_path:
        st.info("Please upload a CSV or enter a valid path to begin.")
        st.stop()

    # Decide input source and, if needed, auto-detect the separator
    file_source = uploaded_file or csv_path
    detected_sep = None
    effective_sep = sep

    if sep is None:
        try:
            if uploaded_file is not None:
                raw_bytes = uploaded_file.getvalue()
            else:
                with open(csv_path, "rb") as f:
                    raw_bytes = f.read(8192)

            try:
                sample_text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                sample_text = raw_bytes.decode("latin-1", errors="ignore")

            detected_sep = detect_separator_from_sample(sample_text, decimal=decimal)
        except Exception:
            detected_sep = None

        if detected_sep:
            effective_sep = detected_sep
            st.sidebar.info(f"Auto-detected separator: {repr(detected_sep)}")
        else:
            effective_sep = None
            st.sidebar.warning(
                "Could not reliably detect the separator. "
                "If the preview looks wrong, please choose a separator manually."
            )
    else:
        effective_sep = sep

    try:
        df = load_csv(
            file_source,
            sep=effective_sep,
            decimal=decimal,
            header=header,
            skiprows=skiprows,
        )
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

    # --- Date parsing options (sidebar) --- #
    dayfirst = False
    date_format = None

    if date_col:
        st.sidebar.subheader("Date parsing")

        dayfirst = st.sidebar.checkbox(
            "Day first (DD/MM instead of MM/DD)",
            value=False,
            help="Enable this if day and month seem inverted",
        )

        date_format = st.sidebar.text_input(
            "Explicit date format (optional)",
            value="",
            help="Example: %d/%m/%Y, %Y-%m-%d, %d.%m.%Y",
        )
        date_format = date_format.strip() or None

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

    # --- Per-series style options (compact rows) --- #
    series_styles = st.session_state["series_styles"]
    active_styles = {}
    enable_secondary_axis = False

    if y_cols:
        st.subheader("🎨 Style options per Y-column")
        dash_map = {
            "Solid": "solid",
            "Dash": "dash",
            "Dot": "dot",
            "Dashdot": "dashdot",
        }
        label_map = {
            "solid": "Solid",
            "dash": "Dash",
            "dot": "Dot",
            "dashdot": "Dashdot",
        }

        enable_secondary_axis = st.checkbox("Enable 2nd axis", value=False)

        if enable_secondary_axis and plot_type == "histogram":
            st.info("Secondary axis is disabled for histograms (uses Axis 1 only).")

        for y in y_cols:
            existing = series_styles.get(y, {})
            default_color = existing.get("color", "#1f77b4")
            default_width = existing.get("width", 2)
            default_dash_value = existing.get("dash", "solid")
            default_dash_label = label_map.get(default_dash_value, "Solid")

            c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 3, 2])

            with c1:
                st.markdown(f"**{y}**")
            with c2:
                color = st.color_picker(
                    "Color",
                    value=default_color,
                    key=f"color_{y}",
                    label_visibility="collapsed",
                )
            with c3:
                width = st.number_input(
                    "Width",
                    min_value=1,
                    max_value=8,
                    value=default_width,
                    step=1,
                    key=f"width_{y}",
                    label_visibility="collapsed",
                )
            with c4:
                dash_label = st.selectbox(
                    "Line type",
                    options=list(dash_map.keys()),
                    index=list(dash_map.keys()).index(default_dash_label),
                    key=f"dash_{y}",
                    label_visibility="collapsed",
                )

            axis_value = (existing.get("axis", 1) if isinstance(existing, dict) else 1)

            with c5:
                if enable_secondary_axis and plot_type != "histogram":
                    axis_choice = st.selectbox(
                        "Axis",
                        options=[1, 2],
                        index=0 if axis_value == 1 else 1,
                        key=f"axis_{y}",
                        label_visibility="collapsed",
                    )
                else:
                    axis_choice = 1

            series_styles[y] = {
                "color": color,
                "width": width,
                "dash": dash_map[dash_label],
                "axis": axis_choice,
            }
            active_styles[y] = series_styles[y]

    # --- Plot theme --- #
    template = st.selectbox(
        "Plotly theme",
        ["plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"],
        index=0,
    )

    # --- Date filters (if date column is present) --- #
    df = parse_dates_flexible(
        df,
        date_col,
        dayfirst=dayfirst,
        date_format=date_format,
    )
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
            start_date_val = st.sidebar.date_input("Start date", value=min_date.date())
            end_date_val = st.sidebar.date_input("End date", value=max_date.date())
            start_date = pd.to_datetime(start_date_val)
            end_date = pd.to_datetime(end_date_val)

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
        fig = None
    else:
        try:
            if date_col and x_col == date_col and plot_type == "line":
                fig = plot_time_series(
                    filtered_df,
                    date_col,
                    y_cols,
                    template=template,
                    styles=active_styles,
                    enable_secondary_axis=enable_secondary_axis,
                )
            else:
                fig = make_figure(
                    filtered_df,
                    x_col,
                    y_cols,
                    plot_type,
                    template=template,
                    styles=active_styles,
                    enable_secondary_axis=enable_secondary_axis,    
                )

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating plot: {e}")
            fig = None

    # --- Save plots --- #
    st.subheader("💾 Save plot configurations")

    plot_label = st.text_input(
        "Optional label for this plot",
        value="",
        help="Give this plot a short name (e.g. 'Winter 2020 hydrograph').",
    )

    if st.button("Save current plot configuration") and fig is not None:
        try:
            label_value = plot_label.strip() or None
            save_plot_config(
                st.session_state["saved_plots"],
                filtered_df,
                x_col,
                y_cols,
                plot_type,
                template,
                label=label_value,
                styles=active_styles,
            )
            st.success("Plot configuration saved!")
        except Exception as e:
            st.error(f"Could not save plot configuration: {e}")

    # --- Saved plots display (with delete option, guarded by try/except) --- #
    if st.session_state["saved_plots"]:
        st.subheader("🗂 Saved plots")
        index_to_delete = None

        for i, p in enumerate(st.session_state["saved_plots"], start=1):
            desc = p.get("label") or f"{', '.join(p['y_cols'])} vs {p['x_col']} ({p['plot_type']})"
            cols_header = st.columns([6, 1])
            with cols_header[0]:
                st.markdown(f"**Saved plot #{i}:** {desc}")
            with cols_header[1]:
                if st.button("Delete", key=f"delete_saved_{i}"):
                    index_to_delete = i - 1

            styles_for_plot = p.get("styles")
            try:
                fig_saved = make_figure(
                    p["data"],
                    p["x_col"],
                    p["y_cols"],
                    p["plot_type"],
                    template=p["template"],
                    title=desc,
                    styles=styles_for_plot,
                    enable_secondary_axis=True,
                )
                st.plotly_chart(fig_saved, use_container_width=True, key=f"saved_plot_{i}")
            except Exception as e:
                st.error(f"Error displaying saved plot #{i}: {e}")

        if index_to_delete is not None:
            st.session_state["saved_plots"].pop(index_to_delete)
            persist_saved_plots_to_disk(st.session_state["saved_plots"])
            st.success("Plot configuration deleted.")

    # --- Compare saved plots (read-only) --- #
    if st.session_state["saved_plots"]:
        st.subheader("🔍 Compare saved plots")

        saved = st.session_state["saved_plots"]
        option_labels = []
        index_by_label = {}
        for i, p in enumerate(saved, start=1):
            desc = p.get("label") or f"{', '.join(p['y_cols'])} vs {p['x_col']} ({p['plot_type']})"
            label = f"#{i}: {desc}"
            option_labels.append(label)
            index_by_label[label] = i - 1

        selected_labels = st.multiselect(
            "Select one or more saved plots to compare",
            options=option_labels,
        )
        selected_indices = [index_by_label[label] for label in selected_labels]

        sync_x = st.checkbox("Synchronize x-axis range", value=True)

        if len(selected_indices) == 0:
            st.info("Select at least one plot above to show a comparison.")

        elif len(selected_indices) == 1:
            idx = selected_indices[0]
            p = saved[idx]
            desc = p.get("label") or f"{', '.join(p['y_cols'])} vs {p['x_col']} ({p['plot_type']})"
            styles_for_plot = p.get("styles")
            fig_cmp = make_figure(
                p["data"],
                p["x_col"],
                p["y_cols"],
                p["plot_type"],
                template=p.get("template", "plotly_white"),
                title=desc,
                styles=styles_for_plot,
                enable_secondary_axis=True,
            )
            st.plotly_chart(fig_cmp, use_container_width=True, key=f"compare_{idx}")

        elif len(selected_indices) == 2 and sync_x:
            # Shared-x subplots (perfect sync when zooming)
            idx1, idx2 = selected_indices
            p1, p2 = saved[idx1], saved[idx2]

            desc1 = p1.get("label") or f"{', '.join(p1['y_cols'])} vs {p1['x_col']} ({p1['plot_type']})"
            desc2 = p2.get("label") or f"{', '.join(p2['y_cols'])} vs {p2['x_col']} ({p2['plot_type']})"

            fig1 = make_figure(
                p1["data"],
                p1["x_col"],
                p1["y_cols"],
                p1["plot_type"],
                template=p1.get("template", "plotly_white"),
                title=desc1,
                styles=p1.get("styles"),
                enable_secondary_axis=True,
            )
            fig2 = make_figure(
                p2["data"],
                p2["x_col"],
                p2["y_cols"],
                p2["plot_type"],
                template=p2.get("template", "plotly_white"),
                title=desc2,
                styles=p2.get("styles"),
                enable_secondary_axis=True,
            )

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.07,
                subplot_titles=(desc1, desc2),
            )

            for tr in fig1.data:
                fig.add_trace(tr, row=1, col=1)
            for tr in fig2.data:
                fig.add_trace(tr, row=2, col=1)

            fig.update_layout(
                margin=dict(l=40, r=20, t=60, b=40),
            )

            st.plotly_chart(fig, use_container_width=True, key="compare_shared")

        elif len(selected_indices) == 2 and not sync_x:
            # Two independent stacked plots (no sync)
            st.markdown("**Stacked comparison (two plots):**")
            for idx in selected_indices:
                p = saved[idx]
                desc = p.get("label") or f"{', '.join(p['y_cols'])} vs {p['x_col']} ({p['plot_type']})"
                styles_for_plot = p.get("styles")
                fig_cmp = make_figure(
                    p["data"],
                    p["x_col"],
                    p["y_cols"],
                    p["plot_type"],
                    template=p.get("template", "plotly_white"),
                    title=desc,
                    styles=styles_for_plot,
                    enable_secondary_axis=True,
                )
                st.plotly_chart(fig_cmp, use_container_width=True, key=f"compare_{idx}")

        else:
            # 3 or more plots
            if sync_x:
                max_plots = len(selected_indices)
                grid_cols = st.number_input(
                    "Number of columns for comparison layout",
                    min_value=1,
                    max_value=max_plots,
                    value=1,
                    step=1,
                    key="compare_cols_grid",
                )
                grid_rows = st.number_input(
                    "Number of rows for comparison layout",
                    min_value=1,
                    max_value=max_plots,
                    value=max_plots,
                    step=1,
                    key="compare_rows_grid",
                )
                total_cells = grid_rows * grid_cols
                if total_cells < max_plots:
                    st.warning(
                        f"Grid has only {total_cells} cells, "
                        f"so only the first {total_cells} selected plots will be shown."
                    )

                # Build per-subplot figures
                subplot_titles = []
                figs = []
                usable_indices = selected_indices[:total_cells]
                for idx in usable_indices:
                    p = saved[idx]
                    desc = p.get("label") or f"{', '.join(p['y_cols'])} vs {p['x_col']} ({p['plot_type']})"
                    subplot_titles.append(desc)
                    figs.append(
                        make_figure(
                            p["data"],
                            p["x_col"],
                            p["y_cols"],
                            p["plot_type"],
                            template=p.get("template", "plotly_white"),
                            title=desc,
                            styles=p.get("styles"),
                            enable_secondary_axis=True,
                        )
                    )

                # Pad titles to match rows*cols
                while len(subplot_titles) < total_cells:
                    subplot_titles.append("")

                fig = make_subplots(
                    rows=grid_rows,
                    cols=grid_cols,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=tuple(subplot_titles),
                )

                for k, f_sub in enumerate(figs):
                    row = k // grid_cols + 1
                    col = k % grid_cols + 1
                    for tr in f_sub.data:
                        fig.add_trace(tr, row=row, col=col)

                # NEW FIX: link *all* x-axes to the main x-axis ("x")
                xaxis_keys = [ax for ax in fig.layout if ax.startswith("xaxis")]
                for ax in xaxis_keys:
                    if ax == "xaxis":
                        continue  # base axis
                    fig.layout[ax].update(matches="x")

                fig.update_layout(
                    margin=dict(l=40, r=20, t=60, b=40),
                )

                st.plotly_chart(fig, use_container_width=True, key="compare_multi_shared")

            else:
                # Comparison grid; no sync
                st.markdown("**Comparison grid:**")
                cols_cmp = st.columns(2)
                for j, idx in enumerate(selected_indices):
                    p = saved[idx]
                    desc = p.get("label") or f"{', '.join(p['y_cols'])} vs {p['x_col']} ({p['plot_type']})"
                    styles_for_plot = p.get("styles")
                    fig_cmp = make_figure(
                        p["data"],
                        p["x_col"],
                        p["y_cols"],
                        p["plot_type"],
                        template=p.get("template", "plotly_white"),
                        title=desc,
                        styles=styles_for_plot,
                        enable_secondary_axis=True,
                    )
                    col = cols_cmp[j % 2]
                    col.plotly_chart(fig_cmp, use_container_width=True, key=f"compare_{idx}")


if __name__ == "__main__":
    main()
