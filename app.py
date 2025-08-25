import pandas as pd
from pathlib import Path

import dash
from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
from dash_bootstrap_templates import load_figure_template


# --------------------------
# Theme & Templates
# --------------------------
load_figure_template("flatly")
px.defaults.template = "flatly"

# --------------------------
# Constants
# --------------------------
LAST_UPDATED = "آخر تحديث: 2025-08-24 23:00"  # <-- hard-coded for now

# --------------------------
# Data Loading
# --------------------------
DATA_PATH = Path(__file__).parent / "data.xlsx"

def load_data(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_excel(path, sheet_name=0)
    except Exception:
        df = pd.DataFrame(columns=[
            "ت.ت", "الجهة", "المنطقة",
            "عدد الحسابات المصرفية", "عدد المواطنين",
            "عدد الحسابات التي لم يتم إدخاله", "نسبة الإدخال"
        ])
    # Normalize column names and trim labels
    df.columns = [c.strip() for c in df.columns]
    for c in ["المنطقة", "الجهة"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Numerics
    for c in ["عدد الحسابات المصرفية", "عدد المواطنين", "عدد الحسابات التي لم يتم إدخاله", "نسبة الإدخال"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Normalize percentage to 0..100
    if "نسبة الإدخال" in df.columns and df["نسبة الإدخال"].max(skipna=True) is not None:
        if df["نسبة الإدخال"].max(skipna=True) <= 1:
            df["نسبة الإدخال"] = df["نسبة الإدخال"] * 100.0

    # If "غير مُدخلة" missing, derive it
    if all(col in df.columns for col in ["عدد المواطنين", "عدد الحسابات المصرفية"]) and "عدد الحسابات التي لم يتم إدخاله" not in df.columns:
        df["عدد الحسابات التي لم يتم إدخاله"] = (df["عدد المواطنين"] - df["عدد الحسابات المصرفية"]).clip(lower=0)

    return df

df_raw = load_data(DATA_PATH)
regions = sorted(df_raw["المنطقة"].dropna().unique()) if "المنطقة" in df_raw.columns else []
orgs = sorted(df_raw["الجهة"].dropna().unique()) if "الجهة" in df_raw.columns else []


# --------------------------
# App Init
# --------------------------
external_stylesheets = [
    dbc.themes.FLATLY,
    "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css",
]
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server
app.title = "لوحة متابعة الإدخال"


# --------------------------
# Small CSS (green active radios)
# --------------------------
app.index_string = app.index_string.replace(
    "</head>",
    """
    <style>
      /* Make active radio (checked) use Bootstrap success green */
      input.radio-green:checked {
        background-color: #198754 !important;
        border-color: #198754 !important;
      }
      /* Slightly stronger card shadows */
      .shadow-custom { box-shadow: 0 .25rem .75rem rgba(0,0,0,.06) !important; }
    </style>
    </head>
    """
)


# --------------------------
# Components
# --------------------------
def kpi_card(title: str, value: str = "—", sub: str = None, id: str = None):
    return dbc.Card(
        dbc.CardBody(
            # inner container holds dynamic content and keeps centering
            html.Div(
                [
                    html.Div(title, className="text-muted"),
                    html.H3(value, className="my-1"),
                    html.Small(sub, className="text-secondary") if sub else None,
                ],
                id=id,  # <-- set id here (NOT on the Card)
                className="d-flex flex-column justify-content-center align-items-center text-center w-100 h-100"
            ),
            className="p-3",  # body padding
        ),
        className="shadow shadow-custom border-0 rounded-3 bg-white h-100 w-100",
        style={"minHeight": "100px"},
    )



controls = dbc.Card(
    dbc.CardBody([
        html.H5("التحكم والتصفية", className="mb-3 text-end"),

        dbc.Label("المنطقة", className="w-100 text-end"),
        dcc.Dropdown(
            options=[{"label": r, "value": r} for r in regions],
            multi=True, placeholder="اختر منطقة", id="filter-region", className="mb-3"
        ),

        dbc.Label("الجهة", className="w-100 text-end"),
        dcc.Dropdown(
            options=[{"label": o, "value": o} for o in orgs],
            multi=True, placeholder="اختر جهة", id="filter-org", className="mb-3"
        ),

        dbc.Label("المؤشر", className="w-100 text-end"),
        dcc.RadioItems(
            id="measure",
            className="mb-3 dbc text-end",
            options=[
                {"label": "نسبة الإدخال %", "value": "rate"},
                {"label": "الحسابات المُدخلة", "value": "entered"},
                {"label": "الحسابات غير المُدخلة", "value": "missing"},
            ],
            value="rate",
            inline=False,
            inputClassName="radio-green ms-1",
            labelClassName="me-2"
        ),

        dbc.Label("الترتيب", className="w-100 text-end"),
        dcc.RadioItems(
            id="sort-order",
            className="mb-3 dbc text-end",
            options=[
                {"label": "تصاعدي", "value": "asc"},   # small -> large
                {"label": "تنازلي", "value": "desc"},   # large -> small
            ],
            value="desc",
            inline=True,
            inputClassName="radio-green ms-1",
            labelClassName="me-2"
        ),

        dbc.Row([
            dbc.Col(dbc.Button("تطبيق التصفية", id="apply-filters", color="primary", className="w-100"), width=6),
            dbc.Col(dbc.Button("إعادة ضبط", id="reset-filters", color="secondary", outline=True, className="w-100"), width=6),
        ], className="g-2 mb-1"),
    ]),
    className="shadow shadow-custom border-0 rounded-3 bg-white"
)

def scroll_card(graph_id: str, title_id: str):
    """Fixed-height card; inner area scrolls vertically. Title is dynamic."""
    return dbc.Card(
        dbc.CardBody([
            html.H5(id=title_id, className="text-end"),
            html.Div(
                dcc.Graph(id=graph_id, config={"displayModeBar": False}),  # disable modebar
                style={"maxHeight": "460px", "overflowY": "auto"}  # scroll inside card
            ),
        ]),
        className="shadow shadow-custom border-0 rounded-3 bg-white"
    )


# --------------------------
# Layout
# --------------------------
app.layout = dbc.Container([
    # Title row with logo + title + last updated (right aligned)
    dbc.Row([
        # Logo on the far left
        dbc.Col(
            html.Img(src="/assets/logo.jpg", height="80px",
                     style={"objectFit": "contain"}),
            width="auto", align="center"
        ),

        # Last update badge
        dbc.Col(
            dbc.Badge(
                [html.Span("🕒 "), LAST_UPDATED],
                color="light",
                text_color="dark",
                pill=True,
                className="px-3 py-2 shadow-sm",
                style={"fontSize": "0.9rem"}
            ),
            width="auto", align="center"
        ),

        # Title block on the right
        dbc.Col([
            html.H2(
                "راتبك لحظي - لوحة متابعة إحالة البيانات المصرفية",
                className="mt-2 text-end"
            )
        ], align="center"),
    ], className="g-3 justify-content-between align-items-center"),

    dbc.Row([
        dbc.Col([
            # KPI row (values populated by callback on initial load)
            dbc.Row([
                dbc.Col(kpi_card("إجمالي المواطنين", id="kpi-citizens"), md=3, className="d-flex"),
                dbc.Col(kpi_card("إجمالي الحسابات المصرفية", id="kpi-accounts"), md=3, className="d-flex"),
                dbc.Col(kpi_card("حسابات غير مُدخلة", id="kpi-missing"), md=3, className="d-flex"),
                dbc.Col(kpi_card("متوسط نسبة الإدخال", id="kpi-rate"), md=3, className="d-flex"),
            ], className="g-3 my-1 flex-row-reverse align-items-stretch"),

            # Charts
            dbc.Row([
                dbc.Col(scroll_card("fig-orgs", "title-orgs"), md=6),
                dbc.Col(scroll_card("fig-region", "title-region"), md=6),
            ], className="g-3 flex-row-reverse"),
        ], md=9),

        # Right controls
        dbc.Col(html.Div(controls, className="sticky-top", style={"top": "1rem"}), md=3),
    ], className="g-3 flex-row-reverse"),

html.Footer(
        dbc.Container(
            html.P(
                [
                    "Developed by ",
                    html.A("Mohamed Elauzei",
                           href="https://www.linkedin.com/in/mohamedelauzei/",
                           target="_blank",
                           className="text-decoration-none fw-bold")
                ],
                className="text-center text-muted my-4"
            )
        ),
        style={"backgroundColor": "#f8f9fa"}
    )
], fluid=True, style={"backgroundColor": "#f8f9fa", "minHeight": "100vh"})


# --------------------------
# Helpers
# --------------------------
def apply_filters(df: pd.DataFrame, regions_sel, orgs_sel):
    out = df.copy()
    if regions_sel:
        out = out[out["المنطقة"].isin(regions_sel)]
    if orgs_sel:
        out = out[out["الجهة"].isin(orgs_sel)]
    return out

def summarize(df: pd.DataFrame):
    total_cit = int(df["عدد المواطنين"].sum()) if "عدد المواطنين" in df.columns else 0
    total_acc = int(df["عدد الحسابات المصرفية"].sum()) if "عدد الحسابات المصرفية" in df.columns else 0
    total_missing = int(df["عدد الحسابات التي لم يتم إدخاله"].sum()) if "عدد الحسابات التي لم يتم إدخاله" in df.columns else 0
    # Weighted نسبة الإدخال = sum(accounts) / sum(citizens) * 100
    mean_rate = (total_acc / total_cit * 100.0) if total_cit > 0 else 0.0
    return total_cit, total_acc, total_missing, mean_rate

def measure_config(key: str):
    """
    Return (value_column, agg_strategy, pretty_name, xaxis_title, is_percentage)
    For 'rate', compute weighted per group: sum(accounts)/sum(citizens)*100
    """
    if key == "rate":
        return ("__weighted_rate__", "weighted", "نسبة الإدخال %", "% الإدخال", True)
    if key == "entered":
        return ("عدد الحسابات المصرفية", "sum", "الحسابات المُدخلة", "عدد الحسابات", False)
    # missing
    return ("عدد الحسابات التي لم يتم إدخاله", "sum", "الحسابات غير المُدخلة", "عدد الحسابات", False)

def group_metric(dff: pd.DataFrame, group_col: str, measure_key: str, sort_order: str):
    val_col, agg, *_ = measure_config(measure_key)
    if agg == "weighted":
        if ("عدد الحسابات المصرفية" not in dff.columns) or ("عدد المواطنين" not in dff.columns):
            return pd.DataFrame({group_col: [], "__weighted_rate__": []})
        grp = dff.groupby(group_col, as_index=False).agg({"عدد الحسابات المصرفية": "sum", "عدد المواطنين": "sum"})
        grp["__weighted_rate__"] = (grp["عدد الحسابات المصرفية"] / grp["عدد المواطنين"] * 100.0).where(grp["عدد المواطنين"] > 0, 0.0)
        out = grp[[group_col, "__weighted_rate__"]]
        sort_col = "__weighted_rate__"
    else:
        if val_col not in dff.columns:
            return pd.DataFrame({group_col: [], val_col: []})
        out = dff.groupby(group_col, as_index=False)[val_col].sum()
        sort_col = val_col

    # Default should be descending (تنازلي) → big to small at the top
    ascending = (sort_order == "desc")  # flipped to match horizontal category visual order
    out = out.sort_values(sort_col, ascending=ascending)
    return out

def truncate_middle_keep_end(s: str, max_chars: int = 30, end_chars: int = 10) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    end_chars = max(4, min(end_chars, max_chars - 5))
    start_chars = max_chars - end_chars - 1  # 1 for ellipsis
    return s[:start_chars] + "…" + s[-end_chars:]

def make_display_labels(full_series: pd.Series, max_chars: int, end_chars: int) -> pd.Series:
    labels = full_series.astype(str).str.strip().apply(lambda s: truncate_middle_keep_end(s, max_chars, end_chars))
    # ensure uniqueness if two labels still collide
    counts = {}
    out = []
    for full, lab in zip(full_series, labels):
        counts[lab] = counts.get(lab, 0) + 1
        if counts[lab] > 1:
            suffix = str(full)[-3:]
            lab = f"{lab}·{suffix}"
        out.append(lab)
    return pd.Series(out, index=full_series.index)

def fmt_compact(x: float, is_pct: bool) -> str:
    if x is None or pd.isna(x):
        return ""
    if is_pct:
        return f"{x:.1f}%"
    n = float(x)
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000.0 or unit == "T":
            s = f"{n:.1f}{unit}".rstrip("0").rstrip(".")
            return s
        n /= 1000.0
    return f"{x:.0f}"


# --------------------------
# Callback
# --------------------------
@app.callback(
    # KPIs
    Output("kpi-citizens", "children"),
    Output("kpi-accounts", "children"),
    Output("kpi-missing", "children"),
    Output("kpi-rate", "children"),
    # Titles
    Output("title-region", "children"),
    Output("title-orgs", "children"),
    # Figures
    Output("fig-region", "figure"),
    Output("fig-orgs", "figure"),
    # Reset dropdown values on reset
    Output("filter-region", "value"),
    Output("filter-org", "value"),

    # Inputs
    Input("apply-filters", "n_clicks"),
    Input("reset-filters", "n_clicks"),

    # State
    State("filter-region", "value"),
    State("filter-org", "value"),
    State("measure", "value"),
    State("sort-order", "value"),
    prevent_initial_call=False,
)
def update_dashboard(apply_clicks, reset_clicks, regions_sel, orgs_sel, measure_key, sort_order):
    triggered_id = (callback_context.triggered[0]["prop_id"].split(".")[0]
                    if callback_context.triggered else None)

    # Handle reset: clear selections
    reset_regions_value = no_update
    reset_orgs_value = no_update
    if triggered_id == "reset-filters":
        regions_sel = []
        orgs_sel = []
        reset_regions_value = []
        reset_orgs_value = []

    dff = apply_filters(df_raw, regions_sel, orgs_sel)

    # ---- KPIs (centered children)
    total_cit, total_acc, total_missing, mean_rate = summarize(dff)
    kpi1 = html.Div([html.Div("إجمالي المواطنين", className="text-muted"), html.H3(f"{total_cit:,}", className="my-1")], className="text-center w-100")
    kpi2 = html.Div([html.Div("إجمالي الحسابات المصرفية", className="text-muted"), html.H3(f"{total_acc:,}", className="my-1")], className="text-center w-100")
    kpi3 = html.Div([html.Div("حسابات غير مُدخلة", className="text-muted"), html.H3(f"{total_missing:,}", className="my-1")], className="text-center w-100")
    kpi4 = html.Div([html.Div("متوسط نسبة الإدخال", className="text-muted"), html.H3(f"{mean_rate:.1f}%", className="my-1")], className="text-center w-100")

    # Measure config & titles
    val_col, agg, pretty, x_title, is_pct = measure_config(measure_key)
    title_region = f"{pretty} — حسب المنطقة"
    title_orgs = f"{pretty} — حسب الجهة"

    # ------- Figure 1: by region (HORIZONTAL)
    if "المنطقة" in dff.columns and len(dff):
        df_region = group_metric(dff, "المنطقة", measure_key, sort_order)
        disp_col = "__weighted_rate__" if agg == "weighted" else val_col
        df_region["y_label"] = make_display_labels(df_region["المنطقة"], max_chars=32, end_chars=10)
        df_region["bar_text"] = df_region[disp_col].apply(lambda v: fmt_compact(v, is_pct))
        height_region = max(520, 30 * len(df_region))

        fig_region = px.bar(
            df_region, x=disp_col, y="y_label", orientation="h",
            labels={"y_label": "المنطقة", disp_col: x_title},
            hover_data={"y_label": False, disp_col: False},
        )
        fig_region.update_traces(
            text=df_region["bar_text"],
            textposition="outside",
            cliponaxis=False,
            customdata=pd.concat([df_region["المنطقة"], df_region[disp_col]], axis=1).values,
            hovertemplate="<b>%{customdata[0]}</b><br>" + x_title + ": " +
                          (" %{customdata[1]:.1f}%" if is_pct else " %{customdata[1]:,.0f}") +
                          "<extra></extra>"
        )
        fig_region.update_layout(
            height=height_region,
            bargap=0.45,
            margin=dict(t=10, r=80, b=40, l=80),
            xaxis_title=x_title,
            yaxis_title=None,
            xaxis=dict(automargin=True,
                       ticksuffix="%" if is_pct else "",
                       tickformat=None if is_pct else "~s"),
            yaxis=dict(
                categoryorder="array",
                categoryarray=df_region["y_label"].tolist(),
                tickfont=dict(size=12),
                automargin=True
            )
        )
    else:
        fig_region = px.bar()

    # ------- Figure 2: by org (HORIZONTAL)
    if "الجهة" in dff.columns and len(dff):
        df_orgs = group_metric(dff, "الجهة", measure_key, sort_order)
        disp_col = "__weighted_rate__" if agg == "weighted" else val_col
        df_orgs["y_label"] = make_display_labels(df_orgs["الجهة"], max_chars=36, end_chars=12)
        df_orgs["bar_text"] = df_orgs[disp_col].apply(lambda v: fmt_compact(v, is_pct))
        height_orgs = max(520, 30 * len(df_orgs))

        fig_orgs = px.bar(
            df_orgs, x=disp_col, y="y_label", orientation="h",
            labels={"y_label": "الجهة", disp_col: x_title},
            hover_data={"y_label": False, disp_col: False},
        )
        fig_orgs.update_traces(
            text=df_orgs["bar_text"],
            textposition="outside",
            cliponaxis=False,
            customdata=pd.concat([df_orgs["الجهة"], df_orgs[disp_col]], axis=1).values,
            hovertemplate="<b>%{customdata[0]}</b><br>" + x_title + ": " +
                          (" %{customdata[1]:.1f}%" if is_pct else " %{customdata[1]:,.0f}") +
                          "<extra></extra>"
        )
        fig_orgs.update_layout(
            height=height_orgs,
            bargap=0.45,
            margin=dict(t=10, r=80, b=40, l=100),
            xaxis_title=x_title,
            yaxis_title=None,
            xaxis=dict(automargin=True,
                       ticksuffix="%" if is_pct else "",
                       tickformat=None if is_pct else "~s"),
            yaxis=dict(
                categoryorder="array",
                categoryarray=df_orgs["y_label"].tolist(),
                tickfont=dict(size=12),
                automargin=True
            )
        )
    else:
        fig_orgs = px.bar()

    return (
        kpi1, kpi2, kpi3, kpi4,
        title_region, title_orgs,
        fig_region, fig_orgs,
        reset_regions_value, reset_orgs_value
    )


if __name__ == "__main__":
    app.run_server(debug=True)
