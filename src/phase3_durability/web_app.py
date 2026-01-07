"""Local web UI for durability exploration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State

from .dataset import build_datasets


@dataclass
class DataCache:
    data_dir: Path
    datasets: dict[str, pd.DataFrame]

    def load_csvs(self, allowed: set[str] | None = None) -> None:
        self.datasets = {}
        if not self.data_dir.exists():
            return
        for path in sorted(self.data_dir.glob("*.csv")):
            name = path.stem
            if allowed is not None and name not in allowed:
                continue
            df = pd.read_csv(path)
            df = _parse_datetimes(df)
            self.datasets[name] = df

    def build_and_export(self, source: str, output_dir: Path, input_root: Path) -> None:
        datasets = build_datasets(input_root, source=source)
        output_dir.mkdir(parents=True, exist_ok=True)
        allowed = set(datasets.keys())
        for path in output_dir.glob("*.csv"):
            if path.stem in allowed:
                path.unlink()
        for name, df in datasets.items():
            if df is None or df.empty:
                continue
            df.to_csv(output_dir / f"{name}.csv", index=False)
        self.load_csvs(allowed=allowed)


def _parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col.endswith("_at"):
            try:
                df[col] = pd.to_datetime(df[col], utc=True)
            except (ValueError, TypeError):
                continue
    return df


def _dataset_options(datasets: dict[str, pd.DataFrame]) -> list[dict[str, str]]:
    return [{"label": name, "value": name} for name in datasets.keys()]


def _column_options(columns: list[str]) -> list[dict[str, str]]:
    return [{"label": col, "value": col} for col in columns]


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["number"]).columns.tolist()


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(exclude=["number"]).columns.tolist()


def _apply_filter(df: pd.DataFrame, expression: str | None) -> tuple[pd.DataFrame, str | None]:
    if not expression:
        return df, None
    try:
        filtered = df.query(expression, engine="python")
        return filtered, None
    except Exception as exc:  # noqa: BLE001
        return df, f"Filter error: {exc}"


def _aggregate(df: pd.DataFrame, x: str, y: str, color: str | None, agg: str) -> pd.DataFrame:
    if agg == "none":
        return df
    group_cols = [col for col in [x, color] if col]
    if not group_cols:
        return df
    grouped = df.groupby(group_cols, dropna=False)[y]
    if agg == "mean":
        result = grouped.mean().reset_index()
    elif agg == "median":
        result = grouped.median().reset_index()
    elif agg == "sum":
        result = grouped.sum().reset_index()
    elif agg == "count":
        result = grouped.count().reset_index(name=y)
    else:
        result = df
    return result


def _build_figure(df: pd.DataFrame, chart_type: str, x: str, y: str | None, color: str | None, agg: str) -> Any:
    if df.empty:
        return px.scatter(title="No data available")

    if chart_type == "histogram":
        return px.histogram(df, x=x, color=color)

    if y is None:
        return px.scatter(df, x=x, title="Select a Y axis")

    if chart_type in {"bar", "line"}:
        df = _aggregate(df, x, y, color, agg)

    if chart_type == "scatter":
        fig = px.scatter(df, x=x, y=y, color=color)
    elif chart_type == "line":
        fig = px.line(df, x=x, y=y, color=color, markers=True)
    elif chart_type == "bar":
        fig = px.bar(df, x=x, y=y, color=color, barmode="group")
    elif chart_type == "box":
        fig = px.box(df, x=x, y=y, color=color)
    elif chart_type == "violin":
        fig = px.violin(df, x=x, y=y, color=color, box=True, points="outliers")
    elif chart_type == "heatmap":
        fig = px.density_heatmap(df, x=x, y=y)
    else:
        fig = px.scatter(df, x=x, y=y, color=color)

    fig.update_layout(height=520, autosize=False)
    return fig


def _preview_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "transcript_id",
        "run_id",
        "created_at",
        "seed_index",
        "seed_name",
        "behavior",
        "strategy",
        "durability_total",
        "severity_total",
        "severity_universal",
        "severity_behavior_specific",
    ]
    columns = [col for col in preferred if col in df.columns]
    if columns:
        return columns
    return df.columns[:12].tolist()


def create_app(data_cache: DataCache, default_dataset: str | None) -> Dash:
    assets_path = Path(__file__).parent / "assets"
    app = Dash(__name__, assets_folder=str(assets_path))
    app.title = "Durability Lab"

    dataset_options = _dataset_options(data_cache.datasets)

    app.layout = html.Div(
        className="app-shell",
        children=[
            html.Header(
                className="app-header",
                children=[
                    html.Div(
                        className="title-block",
                        children=[
                            html.H1("Durability Lab"),
                            html.P("Interactive explorer for durability + severity metrics"),
                        ],
                    ),
                    html.Div(
                        className="header-actions",
                        children=[
                            html.Button("Reload data", id="reload-btn", n_clicks=0),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="app-grid",
                children=[
                    html.Div(
                        className="controls",
                        children=[
                            html.H2("Build a chart"),
                            html.Label("Dataset"),
                            dcc.Dropdown(
                                id="dataset-select",
                                options=dataset_options,
                                value=default_dataset,
                                clearable=False,
                            ),
                            html.Div(className="divider"),
                            html.Label("Chart type"),
                            dcc.Dropdown(
                                id="chart-type",
                                options=[
                                    {"label": label, "value": value}
                                    for label, value in [
                                        ("Scatter", "scatter"),
                                        ("Line", "line"),
                                        ("Bar", "bar"),
                                        ("Box", "box"),
                                        ("Violin", "violin"),
                                        ("Histogram", "histogram"),
                                        ("Heatmap", "heatmap"),
                                    ]
                                ],
                                value="scatter",
                                clearable=False,
                            ),
                            html.Label("X axis"),
                            dcc.Dropdown(id="x-axis"),
                            html.Label("Y axis"),
                            dcc.Dropdown(id="y-axis"),
                            html.Label("Color"),
                            dcc.Dropdown(id="color-axis"),
                            html.Label("Aggregation"),
                            dcc.Dropdown(
                                id="agg-select",
                                options=[
                                    {"label": "None", "value": "none"},
                                    {"label": "Mean", "value": "mean"},
                                    {"label": "Median", "value": "median"},
                                    {"label": "Sum", "value": "sum"},
                                    {"label": "Count", "value": "count"},
                                ],
                                value="none",
                                clearable=False,
                            ),
                            html.Label("Filter (pandas query)", className="label-muted"),
                            dcc.Textarea(
                                id="filter-expression",
                                placeholder="e.g. durability_total < 0.7 and created_at >= '2025-12-24'",
                                className="filter-box",
                            ),
                            html.Div(id="filter-error", className="error-text"),
                        ],
                    ),
                    html.Div(
                        className="preview",
                        children=[
                            html.Div(
                                className="card",
                                children=[
                                    html.H3("Chart"),
                                    dcc.Graph(
                                        id="main-chart",
                                        className="fixed-graph",
                                        config={"displayModeBar": True, "responsive": False},
                                        style={"height": "520px"},
                                    ),
                                ],
                            ),
                            html.Div(
                                className="card",
                                children=[
                                    html.H3("Snapshot"),
                                    html.Div(id="row-count", className="meta"),
                                    html.Div(id="preview-table"),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Store(id="data-refresh", data=0),
        ],
    )

    @app.callback(
        Output("data-refresh", "data"),
        Output("dataset-select", "options"),
        Output("dataset-select", "value"),
        Input("reload-btn", "n_clicks"),
        State("dataset-select", "value"),
    )
    def _reload_data(n_clicks: int, current_value: str | None):
        _ = n_clicks
        data_cache.load_csvs()
        options = _dataset_options(data_cache.datasets)
        if current_value in data_cache.datasets:
            default_value = current_value
        else:
            default_value = options[0]["value"] if options else None
        return n_clicks, options, default_value

    @app.callback(
        Output("x-axis", "options"),
        Output("x-axis", "value"),
        Output("y-axis", "options"),
        Output("y-axis", "value"),
        Output("color-axis", "options"),
        Output("color-axis", "value"),
        Input("dataset-select", "value"),
        Input("data-refresh", "data"),
    )
    def _update_axes(dataset_name: str | None, refresh: int):  # noqa: ARG001
        if not dataset_name or dataset_name not in data_cache.datasets:
            return [], None, [], None, [], None
        df = data_cache.datasets[dataset_name]
        cols = df.columns.tolist()
        numeric_cols = _numeric_columns(df)
        x_default = "created_at" if "created_at" in cols else (cols[0] if cols else None)
        y_default = "durability_total" if "durability_total" in numeric_cols else (numeric_cols[0] if numeric_cols else None)
        return (
            _column_options(cols),
            x_default,
            _column_options(numeric_cols),
            y_default,
            _column_options(cols),
            None,
        )

    @app.callback(
        Output("main-chart", "figure"),
        Output("row-count", "children"),
        Output("preview-table", "children"),
        Output("filter-error", "children"),
        Input("dataset-select", "value"),
        Input("x-axis", "value"),
        Input("y-axis", "value"),
        Input("chart-type", "value"),
        Input("color-axis", "value"),
        Input("agg-select", "value"),
        Input("filter-expression", "value"),
        Input("data-refresh", "data"),
    )
    def _update_chart(
        dataset_name: str | None,
        x_axis: str | None,
        y_axis: str | None,
        chart_type: str,
        color_axis: str | None,
        agg: str,
        filter_expr: str | None,
        refresh: int,  # noqa: ARG001
    ):
        if not dataset_name or dataset_name not in data_cache.datasets:
            return px.scatter(title="No dataset loaded"), "Rows: 0", html.Div(), None

        df = data_cache.datasets[dataset_name]
        if x_axis is None:
            return px.scatter(title="Select an X axis"), f"Rows: {len(df)}", html.Div(), None

        filtered, error = _apply_filter(df, filter_expr)
        fig = _build_figure(filtered, chart_type, x_axis, y_axis, color_axis, agg)
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))

        preview = filtered.head(12)
        preview_cols = _preview_columns(filtered)
        preview = preview[preview_cols] if preview_cols else preview
        table = html.Table(
            className="preview-table",
            children=[
                html.Thead(html.Tr([html.Th(col) for col in preview.columns])),
                html.Tbody(
                    [
                        html.Tr([html.Td(str(value)) for value in row])
                        for row in preview.to_numpy().tolist()
                    ]
                ),
            ],
        )

        return fig, f"Rows: {len(filtered)}", html.Div(table, className="preview-scroll"), error

    return app


def _datasets_for_source(source: str) -> set[str]:
    if source == "eval":
        return {"eval"}
    if source == "both":
        return {"eval", "transcript", "combined"}
    return {"transcript"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Durability Lab web UI")
    parser.add_argument("--data-dir", default="data/scratch/plots", help="Directory with exported CSVs")
    parser.add_argument("--build", action="store_true", help="Build CSVs from scratch before launch")
    parser.add_argument("--source", choices=["transcript", "eval", "both"], default="transcript")
    parser.add_argument("--input-root", default="data/scratch", help="Directory with Petri outputs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    input_root = Path(args.input_root)
    cache = DataCache(data_dir=data_dir, datasets={})
    allowed = _datasets_for_source(args.source)

    if args.build:
        cache.build_and_export(args.source, data_dir, input_root=input_root)
    else:
        cache.load_csvs(allowed=allowed)

    default_dataset = None
    for name in ("transcript", "eval", "combined"):
        if name in cache.datasets:
            default_dataset = name
            break
    if default_dataset is None:
        default_dataset = next(iter(cache.datasets.keys()), None)
    app = create_app(cache, default_dataset)
    app.run(debug=False, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
