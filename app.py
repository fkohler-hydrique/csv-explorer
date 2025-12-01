from typing import Dict, List, Any, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative
import streamlit as st


# --------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------


def init_session_state():
    if "saved_plots" not in st.session_state:
        st.session_state["saved_plots"] = []  # list[dict]
    if "last_uploaded_name" not in st.session_state:
        st.session_state["last_uploaded_name"] = None


def line_dash_options() -> Dict[str, str]:
    return {
        "Solid": "solid",
        "Dash": "dash",
        "Dot": "dot",
        "Dashdot": "dashdot",
        "Long dash": "longdash",
        "Long dashdot": "longdashdot",
    }


def compute_subplot_grid(n: int) -> Tuple[int, int]:
    """Return (rows, cols) for up to 8 plots, reasonably compact."""
    if n <= 1:
        return 1, 1
    if n == 2:
        return 1, 2
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 3, 2
    return 4, 2  # up to 8


def build_single_plot_figure(
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    plot_type: str,
    y_styles: Dict[str, Dict[str, Any]],
    template: str = "plotly_white",
) -> go.Figure:
    """Create a single Plotly figure for one plot configuration."""
    fig = go.Figure()
    x = df[x_col]

    for idx, y in enumerate(y_cols):
        style = y_styles.get(y, {})
        dash = style.get("dash", "solid")
        width = style.get("width", 2)
        color = style.get("color", qualitative.Plotly[idx % len(qualitative.Plotly)])

        if plot_type == "bar":
            fig.add_bar(
                x=x,
                y=df[y],
                name=y,
                marker=dict(color=color),
            )
        else:
            mode = "lines" if plot_type in ["line", "lines"] else "lines+markers"
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df[y],
                    mode=mode,
                    name=y,
                    line=dict(dash=dash, width=width, color=color),
                )
            )

    fig.update_layout(
        template=template,
        xaxis_title=x_col,
        yaxis_title=", ".join(y_cols),
        legend_title_text="Series",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_comparison_figure(
    plot_configs: List[Dict[str, Any]],
    sync_x: bool = True,
) -> go.Figure:
    """
    Build a comparison figure with multiple subplots from saved plot configs.

    Each element in plot_configs is a dict with:
      - 'data' : dataframe
      - 'x_col'
      - 'y_cols'
      - 'plot_type'
      - 'template'
      - 'y_styles'
      - 'label' (optional)
    """
    n = len(plot_configs)
    rows, cols = compute_subplot_grid(n)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=False,  # we will control matching manually
        subplot_titles=[
            cfg.get("label", f"Plot {i + 1}") for i, cfg in enumerate(plot_configs)
        ],
    )

    dash_map = line_dash_options()
    dash_values = list(dash_map.values())

    for i, cfg in enumerate(plot_configs):
        r = i // cols + 1
        c = i % cols + 1

        df = cfg["data"]
        x_col = cfg["x_col"]
        y_cols = cfg["y_cols"]
        plot_type = cfg.get("plot_type", "line")
        y_styles = cfg.get("y_styles", {})

        x = df[x_col]

        for j, y in enumerate(y_cols):
            style = y_styles.get(y, {})
            dash = style.get("dash", dash_values[j % len(dash_values)])
            width = style.get("width", 2)
            color = style.get(
                "color", qualitative.Plotly[j % len(qualitative.Plotly)]
            )

            if plot_type == "bar":
                fig.add_bar(
                    x=x,
                    y=df[y],
                    name=f"{cfg.get('label', f'Plot {i+1}')}: {y}",
                    row=r,
                    col=c,
                )
            else:
                mode = "lines" if plot_type in ["line", "lines"] else "lines+markers"
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=df[y],
                        mode=mode,
                        name=f"{cfg.get('label', f'Plot {i+1}')}: {y}",
                        line=dict(dash=dash, width=width, color=color),
                        showlegend=False,  # keep legend smaller
                    ),
                    row=r,
                    col=c,
                )

        fig.update_xaxes(title_text=x_col, row=r, col=c)
        fig.update_yaxes(title_text=", ".join(y_cols), row=r, col=c)

    # Global x-sync / de-sync
    if sync_x:
        # All x-axes follow the same range
        fig.update_xaxes(matches="x")
    else:
        # Remove matches so that all subplots are independent again
        for ax_name in fig.layout:
            if isinstance(ax_name, str) and ax_name.startswith("xaxis"):
                axis = fig.layout[ax_name]
                if hasattr(axis, "matches"):
                    axis.matches = None

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
        height=max(400, 280 * rows),
    )
    return fig


# --------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="CSV Explorer",
        layout="wide",
    )

    init_session_state()

    st.title("CSV Explorer")

    # ----------------------------------------------------------------
    # Sidebar: upload (kept simple, like before)
    # ----------------------------------------------------------------
    st.sidebar.header("1. Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV file", type=["csv"], key="csv_uploader"
    )

    if uploaded_file is None:
        st.info("Upload a CSV file to get started.")
        return

    df = pd.read_csv(uploaded_file)
    st.session_state["last_uploaded_name"] = uploaded_file.name

    all_cols = list(df.columns)
    if not all_cols:
        st.warning("No columns detected in this CSV.")
        return

    # ----------------------------------------------------------------
    # MAIN AREA: data preview + basic stats (with hide option)
    # ----------------------------------------------------------------
    with st.expander("Data preview & statistics (click to show/hide)", expanded=True):
        st.subheader("Data preview")
        st.dataframe(df.head(100))

        st.markdown("**Shape**")
        st.write(df.shape)

        st.markdown("**Summary statistics**")
        # include='all' to try to match previous behavior (numeric + non-numeric)
        try:
            st.dataframe(df.describe(include="all").transpose())
        except Exception:
            st.dataframe(df.describe().transpose())

    # ----------------------------------------------------------------
    # MAIN AREA: plot selection (x/y) – as you preferred before
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("Plot selection")

    default_x = all_cols[0]
    col_x, col_y = st.columns(2)

    with col_x:
        x_col = st.selectbox(
            "X-axis column",
            options=all_cols,
            index=0 if default_x in all_cols else 0,
            key="x_col_main",
        )

    possible_y = [c for c in all_cols if c != x_col]
    if not possible_y:
        st.warning("Need at least one column other than X to plot.")
        return

    default_y = possible_y[:1]

    with col_y:
        y_cols = st.multiselect(
            "Y-axis column(s)",
            options=possible_y,
            default=default_y,
            key="y_cols_main",
        )

    if not y_cols:
        st.warning("Select at least one Y column.")
        return

    # ----------------------------------------------------------------
    # Sidebar: plot parameters + per-Y styling (minimalistic)
    # ----------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.header("2. Plot configuration")

    plot_type_label = st.sidebar.selectbox(
        "Plot type",
        options=["Line", "Line with markers", "Bar"],
        index=0,
    )
    if plot_type_label == "Bar":
        plot_type = "bar"
    elif plot_type_label == "Line with markers":
        plot_type = "scatter"
    else:
        plot_type = "line"

    template = st.sidebar.selectbox(
        "Plot template",
        options=["plotly_white", "plotly_dark", "simple_white", "ggplot2"],
        index=0,
    )

    # Per-Y styling
    st.sidebar.markdown("---")
    st.sidebar.header("3. Style each Y")

    dash_options = line_dash_options()
    dash_labels = list(dash_options.keys())
    color_palette = qualitative.Plotly

    y_styles: Dict[str, Dict[str, Any]] = {}

    for idx, y in enumerate(y_cols):
        st.sidebar.markdown(f"**{y}**")
        col_w, col_t, col_c = st.sidebar.columns([1, 1, 1])

        with col_w:
            width = st.number_input(
                "Width",
                min_value=1,
                max_value=8,
                value=2,
                step=1,
                key=f"{y}_width",
            )

        with col_t:
            dash_label = st.selectbox(
                "Type",
                options=dash_labels,
                index=0,
                key=f"{y}_dash_style",
            )
            dash_value = dash_options[dash_label]

        with col_c:
            default_color = color_palette[idx % len(color_palette)]
            color = st.color_picker(
                "Color",
                value=default_color,
                key=f"{y}_color",
            )

        y_styles[y] = {"dash": dash_value, "width": width, "color": color}

    # ----------------------------------------------------------------
    # MAIN AREA: current plot
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("Current plot")

    fig = build_single_plot_figure(
        df=df,
        x_col=x_col,
        y_cols=y_cols,
        plot_type=plot_type,
        y_styles=y_styles,
        template=template,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------------------
    # Save plot configuration
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("Saved plots")

    col_save, col_clear = st.columns([1, 1])

    with col_save:
        if st.button("💾 Save current plot configuration"):
            cfg = {
                "data": df.copy(),  # snapshot of the data
                "x_col": x_col,
                "y_cols": y_cols.copy(),
                "plot_type": plot_type,
                "template": template,
                "y_styles": y_styles,
                "label": f"{', '.join(y_cols)} vs {x_col}",
                "source_file": st.session_state["last_uploaded_name"],
            }
            st.session_state["saved_plots"].append(cfg)
            st.success("Plot configuration saved.")

    with col_clear:
        if st.button("🗑️ Clear all saved plots"):
            st.session_state["saved_plots"] = []
            st.info("All saved plots removed.")

    saved_plots: List[Dict[str, Any]] = st.session_state["saved_plots"]

    if not saved_plots:
        st.caption("No saved plots yet.")
        return

    for i, cfg in enumerate(saved_plots):
        label = cfg.get("label", f"Plot {i + 1}")
        source_file = cfg.get("source_file") or "unknown file"
        st.write(
            f"**#{i + 1}** – {label} "
            f"(type: `{cfg.get('plot_type', 'line')}`, from `{source_file}`)"
        )

    # ----------------------------------------------------------------
    # Comparison selection + figure (with working sync)
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("Compare saved plots")

    indices = list(range(len(saved_plots)))
    max_compare = 8

    selected_indices = st.multiselect(
        f"Select plots to compare (up to {max_compare})",
        options=indices,
        default=indices[: min(len(indices), 2)],
        format_func=lambda i: f"#{i + 1}: {saved_plots[i].get('label', f'Plot {i+1}')}",
    )

    if len(selected_indices) > max_compare:
        st.warning(f"Please select at most {max_compare} plots.")
        selected_indices = selected_indices[:max_compare]

    sync_x = st.checkbox(
        "Synchronize x-axis across subplots (zoom one → zoom all)",
        value=True,
    )

    if selected_indices:
        compare_configs = [saved_plots[i] for i in selected_indices]
        compare_fig = build_comparison_figure(compare_configs, sync_x=sync_x)
        st.plotly_chart(compare_fig, use_container_width=True)
    else:
        st.caption("Select at least one saved plot to build a comparison.")


if __name__ == "__main__":
    main()
