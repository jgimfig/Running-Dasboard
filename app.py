"""Running Dashboard — Dash application to analyze Garmin FIT training data."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, ALL, ctx, no_update
from datetime import date, timedelta
from data_loader import (
    load_all_data,
    load_apple_health_data,
    format_pace,
    format_duration,
    compute_hr_zone_distribution,
    weekly_summary,
    load_tag_state,
    set_session_tags,
    add_custom_tag,
    UNTAGGED,
    MAX_TAG_LEN,
    HR_ZONE_LABELS,
    HR_ZONE_COLORS,
    REST_HR,
    karvonen_boundaries,
)

# ---------------------------------------------------------------------------
# Data loading (once at startup)
# ---------------------------------------------------------------------------
print("Loading FIT files…")
sessions_df, laps_df, records_df = load_all_data()
print(f"  {len(sessions_df)} activities | {len(laps_df)} laps | {len(records_df):,} records")

print("Loading Apple Health data…")
hrv_df, rhr_df, vo2_df = load_apple_health_data()
print(f"  {len(hrv_df)} HRV days | {len(rhr_df)} RHR days | {len(vo2_df)} VO2Max records")

GLOBAL_MAX_HR = int(sessions_df["max_hr"].max()) if not sessions_df.empty else 200

PLOTLY_TEMPLATE = "plotly_dark"
CARD_BG = "#16213e"
PAGE_BG = "#0f0e17"
SLIDER_BG = "#0d1b2a"
ACCENT = "#52b788"
ACCENT2 = "#e94560"
TEXT = "#e8e8e4"

# ---------------------------------------------------------------------------
# Date slider setup
# ---------------------------------------------------------------------------
MIN_DATE = sessions_df["start_time"].min().date() if not sessions_df.empty else None
MAX_DATE = sessions_df["start_time"].max().date() if not sessions_df.empty else None
TOTAL_DAYS = (MAX_DATE - MIN_DATE).days if MIN_DATE and MAX_DATE else 0


def _make_slider_marks():
    if not MIN_DATE or not MAX_DATE:
        return {}
    marks = {
        0: {"label": MIN_DATE.strftime("%d/%m/%y"), "style": {"color": "#aaa", "fontSize": "11px"}},
    }
    d = MIN_DATE.replace(day=1)
    d = d.replace(month=d.month % 12 + 1) if d.month < 12 else d.replace(year=d.year + 1, month=1)
    while d <= MAX_DATE:
        offset = (d - MIN_DATE).days
        marks[offset] = {"label": d.strftime("%b '%y"), "style": {"color": "#aaa", "fontSize": "11px"}}
        d = d.replace(month=d.month % 12 + 1) if d.month < 12 else d.replace(year=d.year + 1, month=1)
    marks[TOTAL_DAYS] = {"label": MAX_DATE.strftime("%d/%m/%y"), "style": {"color": "#aaa", "fontSize": "11px"}}
    return marks


SLIDER_MARKS = _make_slider_marks()

DIST_MIN = 0
DIST_MAX = int(sessions_df["distance_km"].max()) + 2 if not sessions_df.empty else 25
DIST_MARKS = {
    v: {"label": f"{v} km", "style": {"color": "#aaa", "fontSize": "11px"}}
    for v in [0, 5, 10, 15, 20, DIST_MAX]
}


def offset_to_date(offset: int):
    return MIN_DATE + timedelta(days=int(offset))


def default_date_range():
    """Slider value covering the previous calendar month up to the latest activity.

    Since the slider ends at the last recorded activity, this means "previous month + current
    month" whenever there is recent data. Falls back to the full history when the data is older.
    """
    if not MIN_DATE:
        return [0, 0]
    prev_month_start = (date.today().replace(day=1) - timedelta(days=1)).replace(day=1)
    start = max((prev_month_start - MIN_DATE).days, 0)
    if start > TOTAL_DAYS:
        start = 0
    return [start, TOTAL_DAYS]


def filter_sessions(slider_value, dist_range, tag_filter=None):
    """Return sessions filtered by date and distance sliders, and optionally by tag.

    The returned frame has a "tags" column (list of labels, empty when the session has none).
    A session matches when it has ANY of the selected tags; UNTAGGED matches sessions without labels.
    An empty/None tag_filter means "all tags".
    """
    start = offset_to_date(slider_value[0])
    end = offset_to_date(slider_value[1])
    mask = (
        (sessions_df["start_time"].dt.date >= start)
        & (sessions_df["start_time"].dt.date <= end)
        & (sessions_df["distance_km"] >= dist_range[0])
        & (sessions_df["distance_km"] <= dist_range[1])
    )
    sdf = sessions_df[mask].copy()
    assignments = load_tag_state()["assignments"]
    sdf["tags"] = [assignments.get(n, []) for n in sdf["name"]]
    if tag_filter:
        wanted = set(tag_filter)
        keep = [(set(t) & wanted) or (UNTAGGED in wanted and not t) for t in sdf["tags"]]
        sdf = sdf[[bool(k) for k in keep]]
    sdf = sdf.reset_index(drop=True)
    sdf["activity_num"] = range(1, len(sdf) + 1)
    return sdf


def filter_all(slider_value, dist_range, tag_filter=None):
    """Return sessions, laps, records filtered by date, distance and tag."""
    sdf = filter_sessions(slider_value, dist_range, tag_filter)
    ldf = laps_df[laps_df["activity"].isin(sdf["name"])].copy()
    rdf = records_df[records_df["activity"].isin(sdf["name"])].copy()
    return sdf, ldf, rdf


# ---------------------------------------------------------------------------
# Helper UI components
# ---------------------------------------------------------------------------

def kpi_card(title: str, value: str, subtitle: str = "", color: str = ACCENT):
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="text-muted mb-1",
                   style={"fontSize": "0.78rem", "letterSpacing": "0.05em", "textTransform": "uppercase"}),
            html.H3(value, style={"color": color, "fontWeight": "700", "margin": "0"}),
            html.Small(subtitle, className="text-muted"),
        ]),
        style={"backgroundColor": CARD_BG, "border": "1px solid #2a2a4a", "borderRadius": "12px"},
        className="h-100",
    )


def section_title(text: str):
    return html.H5(text, style={
        "color": TEXT, "fontWeight": "600", "marginBottom": "0.75rem", "marginTop": "1.25rem",
        "borderLeft": f"4px solid {ACCENT}", "paddingLeft": "10px",
    })


def linear_trend(values):
    """Least-squares trend line over the sample index.

    With fewer than 2 points there is no trend to fit (np.polyfit raises), so the values are
    returned unchanged — which happens easily when a tag filter leaves a single session.
    """
    y = np.asarray(values, dtype=float)
    if len(y) < 2:
        return y
    x = np.arange(len(y))
    return np.poly1d(np.polyfit(x, y, 1))(x)


def empty_fig(msg="Sin datos"):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(color="#888", size=14))
    fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG)
    return fig


def graph_card(figure, height: int = 300, click_id: str = None):
    """Card wrapping an embedded figure. With click_id, clicking a point opens that activity's detail."""
    extra = {"id": {"type": "act-graph", "index": click_id}, "className": "clickable-graph"} if click_id else {}
    return dbc.Card(
        dbc.CardBody(
            dcc.Graph(figure=figure, config={"displayModeBar": False},
                      style={"height": f"{height}px"}, **extra)
        ),
        style={"backgroundColor": CARD_BG, "border": "1px solid #2a2a4a", "borderRadius": "12px"},
    )


def graph_placeholder(fig_id: str, height: int = 300, **graph_kwargs):
    """Card with a named placeholder — figure is set by a separate callback."""
    return dbc.Card(
        dbc.CardBody(
            dcc.Graph(id=fig_id, figure=empty_fig(), config={"displayModeBar": False},
                      style={"height": f"{height}px"}, **graph_kwargs)
        ),
        style={"backgroundColor": CARD_BG, "border": "1px solid #2a2a4a", "borderRadius": "12px"},
    )


# ---------------------------------------------------------------------------
# Tab builders (all accept filtered DataFrames as parameters)
# ---------------------------------------------------------------------------

def build_overview(sdf, rdf):
    if sdf.empty:
        return html.Div("Sin actividades en el período seleccionado.", className="p-4 text-muted")

    total_km = sdf["distance_km"].sum()
    total_min = sdf["duration_min"].sum()
    total_cal = sdf["total_calories"].sum()
    n_acts = len(sdf)
    total_ascent = sdf["total_ascent_m"].dropna().sum() if "total_ascent_m" in sdf.columns else 0
    avg_pace = sdf["avg_pace_min_km"].mean()
    wdf = weekly_summary(sdf)

    # Weekly volume
    fig_weekly = go.Figure()
    if not wdf.empty:
        fig_weekly.add_trace(go.Bar(
            x=wdf["week_start"].dt.strftime("%d %b"),
            y=wdf["distance_km"].round(1),
            marker_color=ACCENT,
            hovertemplate="Semana del %{x}<br><b>%{y} km</b><extra></extra>",
            name="Km semanales",
        ))
        fig_weekly.add_trace(go.Scatter(
            x=wdf["week_start"].dt.strftime("%d %b"),
            y=wdf["distance_km"].rolling(3, min_periods=1).mean().round(1),
            mode="lines", line=dict(color=ACCENT2, width=2, dash="dot"),
            name="Media móvil 3w",
            hovertemplate="%{y:.1f} km<extra>Media 3 sem</extra>",
        ))
    fig_weekly.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1),
        yaxis_title="km",
    )

    # Cumulative km
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=sdf["start_time"], y=sdf["distance_km"].cumsum(),
        mode="lines+markers", line=dict(color=ACCENT, width=2), marker=dict(size=5, color=ACCENT),
        fill="tozeroy", fillcolor="rgba(82,183,136,0.15)",
        hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:.1f} km totales</b><extra>%{text}</extra>",
        text=sdf["name"],
    ))
    fig_cum.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=10, r=10, t=10, b=10), yaxis_title="km acumulados",
    )

    # Pace coloured by HR
    df_v = sdf.dropna(subset=["avg_pace_min_km", "avg_hr"])
    fig_scatter = go.Figure()
    if not df_v.empty:
        fig_scatter.add_trace(go.Scatter(
            x=df_v["start_time"], y=df_v["avg_pace_min_km"],
            mode="markers+lines",
            marker=dict(color=df_v["avg_hr"], colorscale="RdYlGn_r", size=9, showscale=True,
                        colorbar=dict(title="FC media", thickness=10, len=0.7)),
            line=dict(color="rgba(255,255,255,0.15)", width=1),
            hovertemplate="%{x|%d/%m/%Y}<br>Ritmo: %{customdata[0]}/km<br>FC: %{customdata[1]} bpm<extra>%{text}</extra>",
            customdata=np.column_stack([
                [format_pace(p) for p in df_v["avg_pace_min_km"]],
                df_v["avg_hr"].astype(int),
            ]),
            text=df_v["name"],
        ))
    fig_scatter.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Ritmo (min/km)", autorange="reversed",
                   tickvals=[4, 5, 6, 7, 8], ticktext=["4:00", "5:00", "6:00", "7:00", "8:00"]),
    )

    # Weekly elevation gain
    fig_weekly_elev = go.Figure()
    if not wdf.empty and "total_ascent_m" in wdf.columns:
        elev_w = wdf.dropna(subset=["total_ascent_m"])
        if not elev_w.empty:
            fig_weekly_elev.add_trace(go.Bar(
                x=elev_w["week_start"].dt.strftime("%d %b"),
                y=elev_w["total_ascent_m"].round(0),
                marker_color="#8ECAE6",
                hovertemplate="Semana del %{x}<br><b>%{y:.0f} m D+</b><extra></extra>",
                name="D+ semanal",
            ))
            fig_weekly_elev.add_trace(go.Scatter(
                x=elev_w["week_start"].dt.strftime("%d %b"),
                y=elev_w["total_ascent_m"].rolling(3, min_periods=1).mean().round(0),
                mode="lines", line=dict(color=ACCENT2, width=2, dash="dot"),
                name="Media móvil 3w",
                hovertemplate="%{y:.0f} m<extra>Media 3 sem</extra>",
            ))
    fig_weekly_elev.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1),
        yaxis_title="D+ (m)",
    )

    # All-routes map
    fig_map_all = go.Figure()
    gps_all = rdf.dropna(subset=["lat", "lon"])
    if not gps_all.empty:
        parts = []
        for act_name in sdf["name"]:
            act_gps = gps_all[gps_all["activity"] == act_name]
            if not act_gps.empty:
                parts.append(act_gps[["lat", "lon"]])
                parts.append(pd.DataFrame({"lat": [None], "lon": [None]}))
        if parts:
            combined = pd.concat(parts, ignore_index=True)
            fig_map_all.add_trace(go.Scattermap(
                lat=combined["lat"], lon=combined["lon"],
                mode="lines",
                line=dict(width=2, color=ACCENT),
                hoverinfo="skip",
            ))
            lat_lo, lat_hi = gps_all["lat"].quantile([0.1, 0.9])
            lon_lo, lon_hi = gps_all["lon"].quantile([0.1, 0.9])
            center_lat = float((lat_lo + lat_hi) / 2)
            center_lon = float((lon_lo + lon_hi) / 2)
            max_range = max(float(lat_hi - lat_lo), float(lon_hi - lon_lo))
            zoom = max(8, min(14, round(12 - np.log2(max(max_range, 0.001) / 0.05))))
            fig_map_all.update_layout(
                map_style="carto-darkmatter",
                map=dict(center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
                paper_bgcolor=CARD_BG,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
            )
    else:
        fig_map_all = empty_fig("Sin datos GPS")

    pace_str = format_pace(avg_pace) + "/km" if not np.isnan(avg_pace) else "—"

    return dbc.Container([
        dbc.Row([
            dbc.Col(kpi_card("Kilómetros totales", f"{total_km:.1f} km"), md=2),
            dbc.Col(kpi_card("Actividades", str(n_acts)), md=2),
            dbc.Col(kpi_card("Tiempo total", format_duration(total_min * 60), color="#8ECAE6"), md=2),
            dbc.Col(kpi_card("Calorías totales", f"{int(total_cal):,} kcal", color="#F4D03F"), md=2),
            dbc.Col(kpi_card("Desnivel ↑ total", f"{total_ascent:,.0f} m", "ganancia acumulada", "#8ECAE6"), md=2),
            dbc.Col(kpi_card("Ritmo medio", pace_str, "", ACCENT), md=2),
        ], className="g-3 mb-2"),
        dbc.Row([
            dbc.Col([section_title("Volumen semanal (km)"), graph_card(fig_weekly, 240)], md=6),
            dbc.Col([section_title("Desnivel semanal (D+)"), graph_card(fig_weekly_elev, 240)], md=6),
        ], className="g-3"),
        dbc.Row([
            dbc.Col([section_title("Km acumulados"), graph_card(fig_cum, 230, "ov-cum")], md=5),
            dbc.Col([section_title("Ritmo por actividad (color = FC media)"),
                     graph_card(fig_scatter, 230, "ov-pace")], md=7),
        ], className="g-3 mt-0"),
        section_title("Todas las rutas"),
        dbc.Row([dbc.Col(graph_card(fig_map_all, 400))]),
    ], fluid=True, className="pt-3")


def build_progression(sdf):
    if sdf.empty:
        return html.Div("Sin actividades en el período seleccionado.", className="p-4 text-muted")

    def trend_line_fig(df2, y_col, color, y_axis_cfg, name):
        fig = go.Figure()
        df2 = df2.dropna(subset=[y_col]).copy()
        if not df2.empty:
            trend = linear_trend(df2[y_col])
            fig.add_trace(go.Scatter(
                x=df2["start_time"], y=df2[y_col],
                mode="markers", marker=dict(color=color, size=8),
                hovertemplate="%{x|%d/%m/%Y}<br>" + name + ": %{customdata}<extra>%{text}</extra>",
                customdata=[format_pace(v) for v in df2[y_col]] if "pace" in y_col else df2[y_col].round(1),
                text=df2["name"], name=name,
            ))
            fig.add_trace(go.Scatter(
                x=df2["start_time"], y=trend,
                mode="lines", line=dict(color=ACCENT2, dash="dash", width=2),
                name="Tendencia", hoverinfo="skip",
            ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
            margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1),
            yaxis=y_axis_cfg,
        )
        return fig

    pace_yax = dict(title="Ritmo (min/km)", autorange="reversed",
                    tickvals=[4, 4.5, 5, 5.5, 6, 6.5, 7],
                    ticktext=["4:00", "4:30", "5:00", "5:30", "6:00", "6:30", "7:00"])
    hr_yax = dict(title="FC media (bpm)")
    cad_yax = dict(title="Cadencia (spm)")

    fig_pace = trend_line_fig(sdf, "avg_pace_min_km", ACCENT, pace_yax, "Ritmo medio")
    fig_hr = trend_line_fig(sdf, "avg_hr", ACCENT2, hr_yax, "FC media")

    df_cad = sdf[sdf["avg_cadence_spm"].fillna(0) > 0].copy()
    fig_cad = go.Figure()
    if not df_cad.empty:
        fig_cad.add_hrect(y0=170, y1=180, line_width=0, fillcolor="rgba(82,183,136,0.1)",
                          annotation_text="Zona ideal", annotation_position="top right")
        fig_cad.add_trace(go.Scatter(
            x=df_cad["start_time"], y=df_cad["avg_cadence_spm"],
            mode="markers+lines", marker=dict(color="#F4D03F", size=7),
            line=dict(color="rgba(244,208,63,0.4)", width=1),
            hovertemplate="%{x|%d/%m/%Y}<br>Cadencia: %{y:.0f} spm<extra>%{text}</extra>",
            text=df_cad["name"],
        ))
        fig_cad.add_trace(go.Scatter(
            x=df_cad["start_time"], y=linear_trend(df_cad["avg_cadence_spm"]),
            mode="lines", line=dict(color=ACCENT2, dash="dash", width=2),
            name="Tendencia", hoverinfo="skip",
        ))
    fig_cad.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=10, r=10, t=10, b=10), yaxis=cad_yax, showlegend=False,
    )

    df_eff = sdf.dropna(subset=["avg_pace_min_km", "avg_hr"]).copy()
    df_eff = df_eff[df_eff["distance_km"] > 1]
    fig_eff = go.Figure()
    if not df_eff.empty:
        fig_eff.add_trace(go.Scatter(
            x=df_eff["avg_hr"], y=df_eff["avg_pace_min_km"],
            mode="markers",
            marker=dict(color=df_eff["activity_num"], colorscale="Viridis",
                        size=df_eff["distance_km"] * 1.5, sizemin=6, showscale=True,
                        colorbar=dict(title="Actividad #", thickness=10, len=0.7)),
            hovertemplate="%{customdata[2]}<br>FC: %{x} bpm<br>Ritmo: %{customdata[0]}/km"
                          "<br>%{customdata[1]:.1f} km<extra>%{text}</extra>",
            customdata=list(zip([format_pace(p) for p in df_eff["avg_pace_min_km"]],
                                df_eff["distance_km"],
                                df_eff["start_time"].dt.strftime("%d/%m/%Y"))),
            text=df_eff["name"],
        ))
    fig_eff.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=10, r=10, t=10, b=10), xaxis_title="FC media (bpm)",
        yaxis=dict(title="Ritmo (min/km)", autorange="reversed",
                   tickvals=[4, 5, 6, 7], ticktext=["4:00", "5:00", "6:00", "7:00"]),
    )

    return dbc.Container([
        dbc.Row([
            dbc.Col([section_title("Evolución del ritmo"), graph_card(fig_pace, 270, "pr-pace")], md=6),
            dbc.Col([section_title("Evolución de FC media"), graph_card(fig_hr, 270, "pr-hr")], md=6),
        ], className="g-3"),
        dbc.Row([
            dbc.Col([section_title("Evolución de cadencia"), graph_card(fig_cad, 270, "pr-cad")], md=6),
            dbc.Col([section_title("Eficiencia: Ritmo vs FC (tamaño = distancia)"),
                     graph_card(fig_eff, 270, "pr-eff")], md=6),
        ], className="g-3"),
    ], fluid=True, className="pt-3")


def build_hr(sdf, ldf, rdf):
    if sdf.empty:
        return html.Div("Sin actividades en el período seleccionado.", className="p-4 text-muted")

    max_hr = int(sdf["max_hr"].max()) if not sdf["max_hr"].isna().all() else GLOBAL_MAX_HR
    zd = compute_hr_zone_distribution(rdf, max_hr)

    # Zone bar
    fig_zones = go.Figure()
    total_z = zd["minutes"].sum()
    bounds = karvonen_boundaries(max_hr)
    z_ranges = [
        f"< {int(bounds[0])}",
        f"{int(bounds[0])}–{int(bounds[1])}",
        f"{int(bounds[1])}–{int(bounds[2])}",
        f"{int(bounds[2])}–{int(bounds[3])}",
        f"{int(bounds[3])}–{max_hr}",
    ]
    for i, row in zd.iterrows():
        label = f"{row['zone']}<br><sub>{z_ranges[i]} bpm</sub>"
        fig_zones.add_trace(go.Bar(
            x=[label], y=[row["minutes"]],
            marker_color=HR_ZONE_COLORS[i], name=row["zone"],
            hovertemplate=f"<b>{row['zone']}</b><br>{z_ranges[i]} bpm<br>%{{y:.0f}} min<extra></extra>",
        ))
        if total_z > 0:
            fig_zones.add_annotation(
                x=row["zone"], y=row["minutes"],
                text=f"{row['minutes'] / total_z * 100:.0f}%",
                showarrow=False, font=dict(color="white", size=11), yshift=8,
            )
    fig_zones.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=10, r=10, t=10, b=80), showlegend=False,
        yaxis_title="Minutos", xaxis_tickangle=-15,
    )

    # Zone donut
    fig_donut = go.Figure(go.Pie(
        labels=zd["zone"], values=zd["minutes"].round(0), hole=0.5,
        marker_colors=HR_ZONE_COLORS, textinfo="percent",
        hovertemplate="%{label}<br>%{value:.0f} min (%{percent})<extra></extra>",
    ))
    fig_donut.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="v", x=1.0, y=0.5),
    )

    # Max/avg HR per activity
    df_mhr = sdf.dropna(subset=["max_hr", "avg_hr"]).copy()
    fig_maxhr = go.Figure()
    if not df_mhr.empty:
        fig_maxhr.add_trace(go.Scatter(
            x=df_mhr["start_time"], y=df_mhr["max_hr"],
            mode="markers+lines", name="FC máx",
            marker=dict(color=ACCENT2, size=8),
            line=dict(color="rgba(233,69,96,0.3)", width=1),
            hovertemplate="%{x|%d/%m/%Y}<br>FC máx: %{y} bpm<extra>%{text}</extra>",
            text=df_mhr["name"],
        ))
        fig_maxhr.add_trace(go.Scatter(
            x=df_mhr["start_time"], y=linear_trend(df_mhr["max_hr"]),
            mode="lines", name="Tendencia máx",
            line=dict(color=ACCENT2, dash="dash", width=2),
            hoverinfo="skip",
        ))
        fig_maxhr.add_trace(go.Scatter(
            x=df_mhr["start_time"], y=df_mhr["avg_hr"],
            mode="markers+lines", name="FC media",
            marker=dict(color=ACCENT, size=6),
            line=dict(color="rgba(82,183,136,0.3)", width=1),
            hovertemplate="%{x|%d/%m/%Y}<br>FC media: %{y} bpm<extra>%{text}</extra>",
            text=df_mhr["name"],
        ))
        fig_maxhr.add_trace(go.Scatter(
            x=df_mhr["start_time"], y=linear_trend(df_mhr["avg_hr"]),
            mode="lines", name="Tendencia media",
            line=dict(color=ACCENT, dash="dash", width=2),
            hoverinfo="skip",
        ))
    fig_maxhr.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=10, r=10, t=10, b=10), yaxis_title="FC (bpm)",
        legend=dict(orientation="h", y=1.1),
    )

    # HR drift
    drift_data = []
    for act in sdf["name"]:
        act_laps = ldf[ldf["activity"] == act].dropna(subset=["avg_hr"])
        if len(act_laps) >= 2:
            mid = len(act_laps) // 2
            drift_data.append({
                "name": act,
                "start_time": sdf.loc[sdf["name"] == act, "start_time"].iloc[0],
                "hr_drift": act_laps.iloc[mid:]["avg_hr"].mean() - act_laps.iloc[:mid]["avg_hr"].mean(),
            })
    drift_df = pd.DataFrame(drift_data) if drift_data else pd.DataFrame()
    fig_drift = go.Figure()
    if not drift_df.empty:
        fig_drift.add_trace(go.Bar(
            x=drift_df["start_time"].dt.strftime("%d/%m"),
            y=drift_df["hr_drift"].round(1),
            marker_color=[ACCENT2 if v > 5 else ACCENT for v in drift_df["hr_drift"]],
            hovertemplate="%{customdata}<br>Deriva FC: %{y:.1f} bpm<extra>%{text}</extra>",
            customdata=drift_df["start_time"].dt.strftime("%d/%m/%Y"),
            text=drift_df["name"],
        ))
        fig_drift.add_hline(y=5, line_dash="dot", line_color="orange",
                            annotation_text="+5 bpm (alerta)", annotation_position="top right")
    fig_drift.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=10, r=10, t=10, b=60), yaxis_title="Deriva FC (bpm)", xaxis_tickangle=-45,
    )

    z2_min = zd.loc[1, "minutes"] if len(zd) > 1 else 0
    z45_min = zd.loc[zd["zone_num"].isin([4, 5]), "minutes"].sum() if "zone_num" in zd.columns else 0
    avg_hr_val = sdf["avg_hr"].mean()

    bounds = karvonen_boundaries(max_hr)
    z_ranges = [
        f"< {int(bounds[0])} bpm",
        f"{int(bounds[0])}–{int(bounds[1])} bpm",
        f"{int(bounds[1])}–{int(bounds[2])} bpm",
        f"{int(bounds[2])}–{int(bounds[3])} bpm",
        f"{int(bounds[3])}–{max_hr} bpm",
    ]

    return dbc.Container([
        dbc.Row([
            dbc.Col(kpi_card("FC Máxima", f"{max_hr} bpm", f"Reposo: {REST_HR} bpm · Karvonen", ACCENT2), md=3),
            dbc.Col(kpi_card("FC media global", f"{avg_hr_val:.0f} bpm" if not np.isnan(avg_hr_val) else "—", "", ACCENT), md=3),
            dbc.Col(kpi_card("Tiempo Z2 (aeróbico)", f"{z2_min:.0f} min",
                             f"{z_ranges[1]} · {z2_min / total_z * 100:.0f}% del total" if total_z > 0 else "",
                             HR_ZONE_COLORS[1]), md=3),
            dbc.Col(kpi_card("Tiempo Z4+Z5 (intenso)", f"{z45_min:.0f} min",
                             f"Z4: {z_ranges[3]}", HR_ZONE_COLORS[3]), md=3),
        ], className="g-3 mb-2"),
        dbc.Row([
            dbc.Col([section_title("Tiempo por zonas"), graph_card(fig_zones, 280)], md=7),
            dbc.Col([section_title("Proporción de zonas"), graph_card(fig_donut, 280)], md=5),
        ], className="g-3"),
        dbc.Row([
            dbc.Col([section_title("FC media y máxima por actividad"), graph_card(fig_maxhr, 250, "hr-maxavg")], md=7),
            dbc.Col([section_title("Deriva de FC (2ª mitad − 1ª mitad)"),
                     graph_card(fig_drift, 250, "hr-drift")], md=5),
        ], className="g-3"),
    ], fluid=True, className="pt-3")


def build_dynamics(sdf):
    if sdf.empty:
        return html.Div("Sin actividades en el período seleccionado.", className="p-4 text-muted")

    def dyn_fig(col, label, color, ideal=None):
        df2 = sdf.dropna(subset=[col]).copy()
        df2 = df2[df2[col] > 0]
        fig = go.Figure()
        if ideal:
            fig.add_hrect(y0=ideal[0], y1=ideal[1], line_width=0,
                          fillcolor="rgba(82,183,136,0.12)",
                          annotation_text="Rango ideal", annotation_position="top right")
        if not df2.empty:
            fig.add_trace(go.Scatter(
                x=df2["start_time"], y=df2[col], mode="markers",
                marker=dict(color=color, size=8),
                hovertemplate=f"%{{x|%d/%m/%Y}}<br>{label}: %{{y:.1f}}<extra>%{{text}}</extra>",
                text=df2["name"],
            ))
            fig.add_trace(go.Scatter(
                x=df2["start_time"], y=linear_trend(df2[col]),
                mode="lines", line=dict(color="rgba(255,255,255,0.4)", dash="dash", width=2),
                hoverinfo="skip",
            ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
            margin=dict(l=10, r=10, t=10, b=10), showlegend=False, yaxis_title=label,
        )
        return fig

    fig_cad = dyn_fig("avg_cadence_spm", "Cadencia (spm)", "#F4D03F", (170, 180))
    fig_vo = dyn_fig("avg_vertical_osc_mm", "Oscilación vertical (mm)", "#8ECAE6", (60, 100))
    fig_stance = dyn_fig("avg_stance_ms", "Tiempo de apoyo (ms)", "#E67E22", (160, 200))
    fig_power = dyn_fig("avg_power_w", "Potencia (W)", ACCENT)

    df_dyn = sdf.dropna(subset=["avg_cadence_spm", "avg_pace_min_km", "avg_hr"]).copy()
    df_dyn = df_dyn[(df_dyn["avg_cadence_spm"] > 0) & (df_dyn["distance_km"] > 1)]
    fig_cad_pace = go.Figure()
    if not df_dyn.empty:
        fig_cad_pace.add_trace(go.Scatter(
            x=df_dyn["avg_cadence_spm"], y=df_dyn["avg_pace_min_km"], mode="markers",
            marker=dict(color=df_dyn["avg_hr"], colorscale="RdYlGn_r", size=9, showscale=True,
                        colorbar=dict(title="FC", thickness=10)),
            hovertemplate="%{customdata[1]}<br>Cadencia: %{x:.0f} spm<br>Ritmo: %{customdata[0]}/km"
                          "<extra>%{text}</extra>",
            customdata=list(zip([format_pace(p) for p in df_dyn["avg_pace_min_km"]],
                                df_dyn["start_time"].dt.strftime("%d/%m/%Y"))),
            text=df_dyn["name"],
        ))
    fig_cad_pace.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Cadencia (spm)",
        yaxis=dict(title="Ritmo (min/km)", autorange="reversed",
                   tickvals=[4, 5, 6, 7], ticktext=["4:00", "5:00", "6:00", "7:00"]),
    )

    avg_cad = sdf.loc[sdf["avg_cadence_spm"].fillna(0) > 0, "avg_cadence_spm"]
    avg_vo = sdf["avg_vertical_osc_mm"].dropna()
    avg_stance = sdf["avg_stance_ms"].dropna()
    avg_pow = sdf.loc[sdf["avg_power_w"].fillna(0) > 0, "avg_power_w"]

    return dbc.Container([
        dbc.Row([
            dbc.Col(kpi_card("Cadencia media", f"{avg_cad.mean():.0f} spm" if not avg_cad.empty else "—", "Ideal: 170–180 spm", "#F4D03F"), md=3),
            dbc.Col(kpi_card("Oscilación vertical", f"{avg_vo.mean():.1f} mm" if not avg_vo.empty else "—", "Ideal: 60–100 mm", "#8ECAE6"), md=3),
            dbc.Col(kpi_card("Tiempo de apoyo", f"{avg_stance.mean():.0f} ms" if not avg_stance.empty else "—", "Ideal: 160–200 ms", "#E67E22"), md=3),
            dbc.Col(kpi_card("Potencia media", f"{avg_pow.mean():.0f} W" if not avg_pow.empty else "—", "", ACCENT), md=3),
        ], className="g-3 mb-2"),
        dbc.Row([
            dbc.Col([section_title("Cadencia (spm)"), graph_card(fig_cad, 240, "dy-cad")], md=6),
            dbc.Col([section_title("Oscilación vertical (mm)"), graph_card(fig_vo, 240, "dy-vo")], md=6),
        ], className="g-3"),
        dbc.Row([
            dbc.Col([section_title("Tiempo de apoyo (ms)"), graph_card(fig_stance, 240, "dy-stance")], md=6),
            dbc.Col([section_title("Potencia (W)"), graph_card(fig_power, 240, "dy-power")], md=6),
        ], className="g-3"),
        dbc.Row([
            dbc.Col([section_title("Cadencia vs Ritmo (color = FC media)"),
                     graph_card(fig_cad_pace, 280, "dy-cadpace")], md=12),
        ], className="g-3"),
    ], fluid=True, className="pt-3")


def build_fitness():
    """Tab showing Apple Health fitness metrics: HRV, resting HR, VO2Max."""
    if hrv_df.empty and rhr_df.empty and vo2_df.empty:
        return html.Div("No se encontraron datos de Apple Health.", className="p-4 text-muted")

    # KPI values (last available data point)
    hrv_now = float(hrv_df["hrv_7d"].iloc[-1]) if not hrv_df.empty else None
    hrv_prev = float(hrv_df["hrv_7d"].iloc[-8]) if not hrv_df.empty and len(hrv_df) >= 8 else None
    rhr_now = float(rhr_df["resting_hr"].iloc[-1]) if not rhr_df.empty else None
    rhr_start = float(rhr_df["resting_hr"].iloc[0]) if not rhr_df.empty else None
    vo2_now = float(vo2_df["vo2max"].iloc[-1]) if not vo2_df.empty else None
    vo2_start = float(vo2_df["vo2max"].iloc[0]) if not vo2_df.empty else None

    def _delta_str(now, ref, invert=False):
        if now is None or ref is None:
            return ""
        delta = now - ref
        arrow = "▲" if delta > 0 else "▼"
        if invert:
            arrow = "▼" if delta > 0 else "▲"
        return f"{arrow} {abs(delta):.1f} vs semana anterior"

    def _trend_str(now, start, unit="", invert=False):
        if now is None or start is None:
            return ""
        delta = now - start
        arrow = "▲" if delta > 0 else "▼"
        if invert:
            arrow = "▼" if delta > 0 else "▲"
        return f"{arrow} {abs(delta):.1f}{unit} vs inicio del histórico"

    # ── HRV chart ────────────────────────────────────────────────────────────
    fig_hrv = go.Figure()
    if not hrv_df.empty:
        fig_hrv.add_trace(go.Scatter(
            x=pd.concat([hrv_df["day"], hrv_df["day"].iloc[::-1]]),
            y=pd.concat([hrv_df["hrv_hi"], hrv_df["hrv_lo"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(82,183,136,0.08)",
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False,
        ))
        fig_hrv.add_trace(go.Scatter(
            x=hrv_df["day"], y=hrv_df["hrv_ms"],
            mode="lines", name="HRV diario",
            line=dict(color="rgba(82,183,136,0.35)", width=1),
            hovertemplate="%{x|%d/%m/%Y}<br>HRV: %{y:.0f} ms<extra>Mediana diaria</extra>",
        ))
        fig_hrv.add_trace(go.Scatter(
            x=hrv_df["day"], y=hrv_df["hrv_7d"],
            mode="lines", name="Media 7 días",
            line=dict(color=ACCENT, width=2.5),
            hovertemplate="%{x|%d/%m/%Y}<br>HRV 7d: %{y:.0f} ms<extra>Media 7d</extra>",
        ))
        # Overlay running sessions as vertical tick marks
        if not sessions_df.empty:
            # Last activity of each day, so each marker maps to one clickable activity
            day_names = (sessions_df.sort_values("start_time")
                         .assign(date=sessions_df["start_time"].dt.date)
                         .groupby("date")["name"].last())
            runs = hrv_df[hrv_df["day"].dt.date.isin(day_names.index)]
            if not runs.empty:
                fig_hrv.add_trace(go.Scatter(
                    x=runs["day"], y=runs["hrv_7d"],
                    mode="markers", name="Día de entreno",
                    marker=dict(symbol="triangle-down", size=9, color=ACCENT2, opacity=0.8),
                    text=runs["day"].dt.date.map(day_names),
                    hovertemplate="%{x|%d/%m/%Y}<br>HRV: %{y:.0f} ms<extra>%{text}</extra>",
                ))
    fig_hrv.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="HRV SDNN (ms)"),
        legend=dict(orientation="h", y=1.08),
    )

    # ── Resting HR chart ─────────────────────────────────────────────────────
    fig_rhr = go.Figure()
    if not rhr_df.empty:
        fig_rhr.add_trace(go.Scatter(
            x=rhr_df["day"], y=rhr_df["resting_hr"],
            mode="markers", name="FC reposo diaria",
            marker=dict(color="rgba(233,69,96,0.4)", size=5),
            hovertemplate="%{x|%d/%m/%Y}<br>FC reposo: %{y:.0f} bpm<extra></extra>",
        ))
        fig_rhr.add_trace(go.Scatter(
            x=rhr_df["day"], y=rhr_df["rhr_7d"],
            mode="lines", name="Media 7 días",
            line=dict(color=ACCENT2, width=2.5),
            hovertemplate="%{x|%d/%m/%Y}<br>RHR 7d: %{y:.1f} bpm<extra>Media 7d</extra>",
        ))
    fig_rhr.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="FC reposo (bpm)"),
        legend=dict(orientation="h", y=1.08),
    )

    # ── VO2Max chart ─────────────────────────────────────────────────────────
    fig_vo2 = go.Figure()
    if not vo2_df.empty:
        fig_vo2.add_trace(go.Scatter(
            x=vo2_df["day"], y=vo2_df["vo2max"],
            mode="lines+markers", name="VO2Max",
            line=dict(color="#8ECAE6", width=2),
            marker=dict(color="#8ECAE6", size=6),
            hovertemplate="%{x|%d/%m/%Y}<br>VO2Max: %{y:.1f} mL/min·kg<extra></extra>",
        ))
        if len(vo2_df) >= 4:
            x_num = np.arange(len(vo2_df))
            z = np.polyfit(x_num, vo2_df["vo2max"], 1)
            fig_vo2.add_trace(go.Scatter(
                x=vo2_df["day"], y=np.poly1d(z)(x_num),
                mode="lines", name="Tendencia",
                line=dict(color=ACCENT2, dash="dash", width=2),
                hoverinfo="skip",
            ))
    fig_vo2.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="VO2Max (mL/min·kg)"),
        legend=dict(orientation="h", y=1.08),
    )

    hrv_str = f"{hrv_now:.0f} ms" if hrv_now else "—"
    rhr_str = f"{rhr_now:.0f} bpm" if rhr_now else "—"
    vo2_str = f"{vo2_now:.1f}" if vo2_now else "—"

    return dbc.Container([
        dbc.Row([
            dbc.Col(kpi_card("HRV (media 7d)", hrv_str,
                             _delta_str(hrv_now, hrv_prev), ACCENT), md=3),
            dbc.Col(kpi_card("FC reposo actual", rhr_str,
                             _trend_str(rhr_now, rhr_start, " bpm", invert=True), ACCENT2), md=3),
            dbc.Col(kpi_card("VO2Max estimado", f"{vo2_str} mL/min·kg",
                             _trend_str(vo2_now, vo2_start, " mL/min·kg"), "#8ECAE6"), md=3),
            dbc.Col(kpi_card("Histórico Apple Health",
                             f"{len(hrv_df)} días",
                             f"{hrv_df['day'].min().strftime('%d/%m/%Y') if not hrv_df.empty else '—'} → hoy",
                             "#F4D03F"), md=3),
        ], className="g-3 mb-2"),
        section_title("Variabilidad de FC (HRV SDNN) — con entrenamientos marcados"),
        dbc.Row([dbc.Col(graph_card(fig_hrv, 280, "fit-hrv"))]),
        dbc.Row([
            dbc.Col([section_title("FC en reposo"), graph_card(fig_rhr, 250)], md=6),
            dbc.Col([section_title("VO2Max estimado (Apple Watch)"), graph_card(fig_vo2, 250)], md=6),
        ], className="g-3"),
    ], fluid=True, className="pt-3")


def build_tags_table(sdf):
    """Table with one multi-select tag dropdown per session (newest first)."""
    if sdf.empty:
        return html.Div("Sin actividades en el período seleccionado.", className="p-4 text-muted")

    state = load_tag_state()
    options = [{"label": t, "value": t} for t in state["options"]]
    rows = []
    for _, r in sdf.sort_values("start_time", ascending=False).iterrows():
        rows.append(html.Tr([
            html.Td(r["start_time"].strftime("%d/%m/%Y %H:%M")),
            html.Td(r["name"]),
            html.Td(f"{r['distance_km']:.1f} km"),
            html.Td(format_pace(r["avg_pace_min_km"]) + "/km"),
            html.Td(dcc.Dropdown(
                id={"type": "tag-select", "index": r["name"]},
                options=options,
                value=state["assignments"].get(r["name"], []),
                multi=True,
                clearable=True,
                placeholder=UNTAGGED,
            ), style={"minWidth": "240px"}),
        ], style={"verticalAlign": "middle"}))
    return dbc.Table(
        [html.Thead(html.Tr([html.Th(h) for h in ["Fecha", "Actividad", "Dist.", "Ritmo", "Etiquetas"]])),
         html.Tbody(rows)],
        striped=True, bordered=False, hover=True, size="sm",
        style={"color": TEXT, "backgroundColor": CARD_BG, "borderRadius": "8px"},
    )


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Running Dashboard",
)
server = app.server  # WSGI entry point for gunicorn (Docker)

_summary_text = (
    f"{len(sessions_df)} actividades · {sessions_df['distance_km'].sum():.0f} km · "
    f"{MIN_DATE.strftime('%d/%m/%Y')} → {MAX_DATE.strftime('%d/%m/%Y')}"
    if not sessions_df.empty else "Sin datos"
)

app.layout = dbc.Container([

    # Hidden metadata element — read by date_transform.js to convert slider offsets to dates
    html.Div(id="slider-meta",
             **{"data-min-date": MIN_DATE.isoformat() if MIN_DATE else ""},
             style={"display": "none"}),

    # Bumped when the set of available tags changes (new custom tag)
    dcc.Store(id="tag-options-version", data=0),
    # Bumped when session tags change while a tag filter is active (views must be recomputed)
    dcc.Store(id="tags-version", data=0),

    # ── Header ──────────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(html.Div([
            html.H2("🏃 Running Dashboard",
                    style={"color": ACCENT, "fontWeight": "700", "marginBottom": "0"}),
            html.Small(_summary_text, style={"color": "#888"}),
        ]), md=12),
    ], className="py-3"),

    # ── Date range filter ───────────────────────────────────────────────────
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("Período: ",
                                  style={"color": "#aaa", "fontSize": "0.82rem", "fontWeight": "500"}),
                        html.Span(id="date-range-display",
                                  style={"color": ACCENT, "fontWeight": "700", "fontSize": "0.82rem"}),
                    ], className="mb-3"),
                    dcc.RangeSlider(
                        id="date-slider",
                        min=0,
                        max=TOTAL_DAYS,
                        value=default_date_range(),  # previous + current month
                        marks=SLIDER_MARKS,
                        tooltip={"always_visible": False, "placement": "bottom"},
                        updatemode="mouseup",
                        className="mb-1",
                    ),
                ], md=6),
                dbc.Col([
                    html.Div([
                        html.Span("Distancia: ",
                                  style={"color": "#aaa", "fontSize": "0.82rem", "fontWeight": "500"}),
                        html.Span(id="distance-range-display",
                                  style={"color": ACCENT, "fontWeight": "700", "fontSize": "0.82rem"}),
                    ], className="mb-3"),
                    dcc.RangeSlider(
                        id="distance-slider",
                        min=DIST_MIN,
                        max=DIST_MAX,
                        step=1,
                        value=[DIST_MIN, DIST_MAX],
                        marks=DIST_MARKS,
                        tooltip={"always_visible": False, "placement": "bottom"},
                        updatemode="mouseup",
                        className="mb-1",
                    ),
                ], md=3),
                dbc.Col([
                    html.Div(html.Span("Etiquetas: ",
                                       style={"color": "#aaa", "fontSize": "0.82rem", "fontWeight": "500"}),
                             className="mb-2"),
                    dcc.Dropdown(
                        id="tag-filter",
                        options=[{"label": t, "value": t} for t in load_tag_state()["options"] + [UNTAGGED]],
                        value=[],
                        multi=True,
                        placeholder="Todas",
                    ),
                ], md=3),
            ]),
        ], style={"padding": "1rem 1rem 2.5rem 1rem"}),
    ], style={
        "backgroundColor": SLIDER_BG,
        "border": "1px solid #2a2a4a",
        "borderRadius": "12px",
        "marginBottom": "1rem",
    }),

    # ── Tabs ────────────────────────────────────────────────────────────────
    dbc.Tabs([
        dbc.Tab(
            dcc.Loading(html.Div(id="tab-overview-content"), color=ACCENT, type="dot"),
            label="Resumen", tab_id="tab-overview",
        ),
        dbc.Tab(
            dcc.Loading(html.Div(id="tab-progression-content"), color=ACCENT, type="dot"),
            label="Progresión", tab_id="tab-progression",
        ),
        dbc.Tab(
            dcc.Loading(html.Div(id="tab-hr-content"), color=ACCENT, type="dot"),
            label="Frecuencia Cardíaca", tab_id="tab-hr",
        ),
        dbc.Tab(
            dcc.Loading(html.Div(id="tab-dynamics-content"), color=ACCENT, type="dot"),
            label="Dinámica de Carrera", tab_id="tab-dynamics",
        ),
        dbc.Tab(
            dcc.Loading(html.Div(id="tab-fitness-content"), color=ACCENT, type="dot"),
            label="Forma Física", tab_id="tab-fitness",
        ),
        dbc.Tab(
            dbc.Container([
                dbc.Row([dbc.Col([
                    html.Label("Selecciona actividad:",
                               style={"color": TEXT, "fontWeight": "600"}),
                    dcc.Dropdown(
                        id="activity-selector",
                        clearable=False,
                        style={"backgroundColor": "#1a1a3e", "color": TEXT},
                        className="mb-3",
                    ),
                ], md=12)]),
                html.Div(id="detail-kpis"),
                # GPS track (distance → lat/lon) used by the client-side hover cursor
                dcc.Store(id="detail-track"),
                dcc.Store(id="detail-cursor-dummy"),
                dbc.Row([
                    dbc.Col([
                        section_title("Mapa del recorrido"),
                        graph_placeholder("detail-map", 430, clear_on_unhover=True),
                    ], md=12),
                ], className="g-3"),
                dbc.Row([
                    dbc.Col([
                        section_title("FC, Ritmo y Altimetría"),
                        graph_placeholder("detail-hr-pace", 380, clear_on_unhover=True),
                    ], md=12),
                ], className="g-3"),
                dbc.Row([
                    dbc.Col([section_title("Frecuencia cardíaca"),
                             graph_placeholder("detail-hr", 220, clear_on_unhover=True)], md=6),
                    dbc.Col([section_title("Ritmo"),
                             graph_placeholder("detail-pace", 220, clear_on_unhover=True)], md=6),
                ], className="g-3"),
                dbc.Row([
                    dbc.Col([section_title("Altimetría"),
                             graph_placeholder("detail-altitude", 200, clear_on_unhover=True)], md=6),
                    dbc.Col([section_title("D+ acumulado por km"),
                             graph_placeholder("detail-cum-elevation", 200, clear_on_unhover=True)], md=6),
                ], className="g-3"),
                dbc.Row([
                    dbc.Col([section_title("Potencia y Cadencia"),
                             graph_placeholder("detail-power-cad", 220, clear_on_unhover=True)], md=12),
                ], className="g-3"),
                dbc.Row([dbc.Col([section_title("Laps"), html.Div(id="detail-laps-table")], md=12)]),
            ], fluid=True, className="pt-3"),
            label="Detalle Actividad", tab_id="tab-detail",
        ),
        dbc.Tab(
            dbc.Container([
                html.P("Asigna una o varias etiquetas a cada sesión y filtra por ellas desde la barra superior. "
                       "Se listan las sesiones del período y rango de distancia seleccionados. "
                       "Los cambios se guardan al momento.",
                       className="text-muted mb-3", style={"fontSize": "0.9rem"}),
                dbc.Row([
                    dbc.Col(dbc.Input(id="new-tag-input", type="text", maxLength=MAX_TAG_LEN,
                                      placeholder="Nueva etiqueta…", debounce=False), md=3),
                    dbc.Col(dbc.Button("Añadir etiqueta", id="add-tag-btn", color="success",
                                       outline=True), md="auto"),
                    dbc.Col(html.Span(id="new-tag-msg", style={"color": "#E67E22", "fontSize": "0.85rem"}),
                            className="d-flex align-items-center"),
                ], className="g-2 mb-3"),
                dcc.Loading(html.Div(id="tab-tags-content"), color=ACCENT, type="dot"),
            ], fluid=True, className="pt-3"),
            label="Etiquetas", tab_id="tab-tags",
        ),
    ], id="main-tabs", active_tab="tab-overview"),

], fluid=True, style={"backgroundColor": PAGE_BG, "minHeight": "100vh", "padding": "0 1.5rem"})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("date-range-display", "children"),
    Output("distance-range-display", "children"),
    Input("date-slider", "value"),
    Input("date-slider", "drag_value"),
    Input("distance-slider", "value"),
    Input("tag-filter", "value"),
    Input("tags-version", "data"),
)
def update_date_label(slider_value, drag_value, dist_range, tag_filter, _tags_version):
    # Use drag_value for live date display while dragging; fall back to value
    live = drag_value if drag_value is not None else slider_value
    start = offset_to_date(live[0])
    end = offset_to_date(live[1])
    # Only count activities when the slider has been released (value, not drag)
    sdf = filter_sessions(slider_value, dist_range, tag_filter)
    n = len(sdf)
    total_km = sdf["distance_km"].sum()
    dist_label = f"{dist_range[0]}–{dist_range[1]} km"
    return (
        f"{start.strftime('%d/%m/%Y')} → {end.strftime('%d/%m/%Y')}  ·  {n} actividades  ·  {total_km:.0f} km",
        dist_label,
    )


@app.callback(
    Output("tab-overview-content", "children"),
    Input("date-slider", "value"),
    Input("distance-slider", "value"),
    Input("tag-filter", "value"),
    Input("tags-version", "data"),
)
def update_overview(slider_value, dist_range, tag_filter, _tags_version):
    sdf, ldf, rdf = filter_all(slider_value, dist_range, tag_filter)
    return build_overview(sdf, rdf)


@app.callback(
    Output("tab-progression-content", "children"),
    Input("date-slider", "value"),
    Input("distance-slider", "value"),
    Input("tag-filter", "value"),
    Input("tags-version", "data"),
)
def update_progression(slider_value, dist_range, tag_filter, _tags_version):
    return build_progression(filter_sessions(slider_value, dist_range, tag_filter))


@app.callback(
    Output("tab-hr-content", "children"),
    Input("date-slider", "value"),
    Input("distance-slider", "value"),
    Input("tag-filter", "value"),
    Input("tags-version", "data"),
)
def update_hr(slider_value, dist_range, tag_filter, _tags_version):
    sdf, ldf, rdf = filter_all(slider_value, dist_range, tag_filter)
    return build_hr(sdf, ldf, rdf)


@app.callback(
    Output("tab-dynamics-content", "children"),
    Input("date-slider", "value"),
    Input("distance-slider", "value"),
    Input("tag-filter", "value"),
    Input("tags-version", "data"),
)
def update_dynamics(slider_value, dist_range, tag_filter, _tags_version):
    return build_dynamics(filter_sessions(slider_value, dist_range, tag_filter))


@app.callback(
    Output("tab-fitness-content", "children"),
    Input("date-slider", "value"),
    Input("distance-slider", "value"),
)
def update_fitness(_slider_value, _dist_range):
    return build_fitness()


@app.callback(
    Output("tab-tags-content", "children"),
    Input("date-slider", "value"),
    Input("distance-slider", "value"),
    Input("tag-options-version", "data"),
)
def update_tags_table(slider_value, dist_range, _options_version):
    # Deliberately ignores the tag filter (and tag changes): rows must not vanish while being edited
    return build_tags_table(filter_sessions(slider_value, dist_range))


@app.callback(
    Output("tags-version", "data"),
    Input({"type": "tag-select", "index": ALL}, "value"),
    State({"type": "tag-select", "index": ALL}, "id"),
    State("tag-filter", "value"),
    State("tags-version", "data"),
    prevent_initial_call=True,
)
def save_session_tags(values, ids, tag_filter, version):
    """Persist edited tags. Only real changes are written, so rebuilding the table is a no-op."""
    current = load_tag_state()["assignments"]
    changes = {
        i["index"]: list(v or [])
        for i, v in zip(ids, values)
        if list(v or []) != current.get(i["index"], [])
    }
    if not changes:
        return no_update
    set_session_tags(changes)
    # The other tabs only depend on tags while a tag filter is active
    return version + 1 if tag_filter else no_update


@app.callback(
    Output("tag-options-version", "data"),
    Output("new-tag-input", "value"),
    Output("new-tag-msg", "children"),
    Input("add-tag-btn", "n_clicks"),
    Input("new-tag-input", "n_submit"),
    State("new-tag-input", "value"),
    State("tag-options-version", "data"),
    prevent_initial_call=True,
)
def add_tag(_clicks, _submits, label, version):
    new_label, error = add_custom_tag(label)
    if error:
        return no_update, no_update, error
    return version + 1, "", f"Etiqueta «{new_label}» añadida."


@app.callback(
    Output("tag-filter", "options"),
    Input("tag-options-version", "data"),
)
def update_tag_filter_options(_options_version):
    return [{"label": t, "value": t} for t in load_tag_state()["options"] + [UNTAGGED]]


@app.callback(
    Output("activity-selector", "options"),
    Output("activity-selector", "value"),
    Input("date-slider", "value"),
    Input("distance-slider", "value"),
    Input("tag-filter", "value"),
    Input("tags-version", "data"),
)
def update_activity_options(slider_value, dist_range, tag_filter, _tags_version):
    sdf = filter_sessions(slider_value, dist_range, tag_filter)
    options = [
        {
            "label": f"{r['start_time'].strftime('%d/%m/%Y %H:%M')} — {r['name']} ({r['distance_km']:.1f} km)",
            "value": r["name"],
        }
        for _, r in sdf.sort_values("start_time", ascending=False).iterrows()
    ]
    value = options[0]["value"] if options else None
    return options, value


@app.callback(
    Output("main-tabs", "active_tab"),
    Output("activity-selector", "options", allow_duplicate=True),
    Output("activity-selector", "value", allow_duplicate=True),
    Input({"type": "act-graph", "index": ALL}, "clickData"),
    State("activity-selector", "options"),
    prevent_initial_call=True,
)
def open_clicked_activity(_click_data, options):
    """Jump to the detail tab for the activity whose point was clicked in any chart."""
    click = ctx.triggered[0]["value"] if ctx.triggered else None
    if not click or not click.get("points"):
        return no_update, no_update, no_update
    name = click["points"][0].get("text")
    if name not in set(sessions_df["name"]):
        # Clicked a trend line or a non-activity trace
        return no_update, no_update, no_update

    options = options or []
    if not any(o["value"] == name for o in options):
        # Activity outside the current filters: add it so the dropdown can show it
        r = sessions_df[sessions_df["name"] == name].iloc[0]
        options = [{
            "label": f"{r['start_time'].strftime('%d/%m/%Y %H:%M')} — {r['name']} ({r['distance_km']:.1f} km)",
            "value": r["name"],
        }] + options
    return "tab-detail", options, name


@app.callback(
    Output("detail-kpis", "children"),
    Output("detail-laps-table", "children"),
    Output("detail-hr-pace", "figure"),
    Output("detail-hr", "figure"),
    Output("detail-pace", "figure"),
    Output("detail-altitude", "figure"),
    Output("detail-cum-elevation", "figure"),
    Output("detail-power-cad", "figure"),
    Output("detail-map", "figure"),
    Output("detail-track", "data"),
    Input("activity-selector", "value"),
)
def update_detail(selected_activity):
    if not selected_activity or sessions_df.empty:
        e = empty_fig()
        return html.Div(), html.Div(), e, e, e, e, e, e, e, None

    row = sessions_df[sessions_df["name"] == selected_activity]
    if row.empty:
        e = empty_fig("Actividad no encontrada")
        return html.Div(), html.Div(), e, e, e, e, e, e, e, None

    row = row.iloc[0]
    act_laps = laps_df[laps_df["activity"] == selected_activity].copy()
    act_records = records_df[records_df["activity"] == selected_activity].copy()

    asc_v = row.get("total_ascent_m")
    desc_v = row.get("total_descent_m")
    asc_str = f"+{int(asc_v)} m" if pd.notna(asc_v) and (asc_v or 0) > 0 else "—"
    desc_str = f"-{int(desc_v)} m" if pd.notna(desc_v) and (desc_v or 0) > 0 else "—"

    kpis = dbc.Container([
        dbc.Row([
            dbc.Col(kpi_card("Distancia", f"{row['distance_km']:.2f} km"), md=2),
            dbc.Col(kpi_card("Tiempo", format_duration(row["duration_sec"])), md=2),
            dbc.Col(kpi_card("Ritmo medio", format_pace(row["avg_pace_min_km"]) + "/km", "", ACCENT), md=2),
            dbc.Col(kpi_card("FC media / máx",
                             f"{int(row['avg_hr'])} / {int(row['max_hr']) if pd.notna(row.get('max_hr')) else '—'} bpm"
                             if pd.notna(row.get("avg_hr")) else "—", "", ACCENT2), md=2),
            dbc.Col(kpi_card("Calorías",
                             f"{int(row['total_calories'])} kcal"
                             if pd.notna(row.get("total_calories")) else "—", "", "#F4D03F"), md=2),
            dbc.Col(kpi_card("Cadencia media",
                             f"{row['avg_cadence_spm']:.0f} spm"
                             if pd.notna(row.get("avg_cadence_spm")) and (row.get("avg_cadence_spm") or 0) > 0 else "—",
                             "", "#8ECAE6"), md=2),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(kpi_card("Desnivel ↑", asc_str, "ganancia de altitud", "#8ECAE6"), md=2),
            dbc.Col(kpi_card("Desnivel ↓", desc_str, "pérdida de altitud", "#E67E22"), md=2),
        ], className="g-2 mt-2"),
    ], fluid=True, className="mb-3")

    if not act_laps.empty:
        rows = []
        # Elapsed time at the end of each lap
        act_laps = act_laps.sort_values("lap_num")
        cum_time = act_laps["duration_sec"].cumsum()
        for (_, lap), elapsed in zip(act_laps.iterrows(), cum_time):
            rows.append(html.Tr([
                html.Td(f"Km {int(lap['lap_num'])}"),
                html.Td(f"{lap['distance_km']:.2f} km"),
                html.Td(format_duration(elapsed)),
                html.Td(format_pace(lap["avg_pace_min_km"]) + "/km"),
                html.Td(f"{int(lap['avg_hr'])} bpm" if pd.notna(lap.get("avg_hr")) else "—"),
                html.Td(f"{int(lap['max_hr'])} bpm" if pd.notna(lap.get("max_hr")) else "—"),
                html.Td(f"{int(lap['avg_cadence_spm'])} spm"
                        if pd.notna(lap.get("avg_cadence_spm")) and (lap.get("avg_cadence_spm") or 0) > 0 else "—"),
                html.Td(f"{int(lap['avg_power_w'])} W"
                        if pd.notna(lap.get("avg_power_w")) and (lap.get("avg_power_w") or 0) > 0 else "—"),
                html.Td(f"+{int(lap['total_ascent_m'])} m"
                        if pd.notna(lap.get("total_ascent_m")) and (lap.get("total_ascent_m") or 0) > 0 else "—"),
                html.Td(f"-{int(lap['total_descent_m'])} m"
                        if pd.notna(lap.get("total_descent_m")) and (lap.get("total_descent_m") or 0) > 0 else "—"),
            ]))
        laps_table = dbc.Table(
            [html.Thead(html.Tr([html.Th(h) for h in
                                 ["Lap", "Dist.", "Tiempo", "Ritmo", "FC media", "FC máx", "Cadencia", "Potencia", "D+", "D-"]])),
             html.Tbody(rows)],
            striped=True, bordered=False, hover=True, size="sm",
            style={"color": TEXT, "backgroundColor": CARD_BG, "borderRadius": "8px"},
        )
    else:
        laps_table = html.P("Sin laps disponibles.", className="text-muted")

    lap_marks = act_laps["distance_km"].cumsum().iloc[:-1] if not act_laps.empty else []
    rec = act_records.dropna(subset=["heart_rate"]) if not act_records.empty else act_records
    rec = rec[rec["heart_rate"] > 0] if not rec.empty else rec
    rec_pace = act_records.dropna(subset=["pace_min_km"]) if not act_records.empty else act_records
    rec_pace = rec_pace[rec_pace["pace_min_km"].between(3, 12)] if not rec_pace.empty else rec_pace
    pace_ticks = dict(tickvals=[4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8],
                      ticktext=["4:00", "4:30", "5:00", "5:30", "6:00", "6:30", "7:00", "7:30", "8:00"])

    # Combined chart: HR + pace, with the elevation profile as a background area
    fig_hr_pace = go.Figure()
    if not act_records.empty:
        alt_bg = act_records.dropna(subset=["altitude_m"])
        if not alt_bg.empty:
            a_lo, a_hi = float(alt_bg["altitude_m"].min()), float(alt_bg["altitude_m"].max())
            a_span = max(a_hi - a_lo, 10)
            fig_hr_pace.add_trace(go.Scatter(
                x=alt_bg["distance_km"], y=alt_bg["altitude_m"],
                mode="lines", name="Altitud", yaxis="y3", fill="tozeroy",
                line=dict(color="rgba(142,202,230,0.5)", width=1), fillcolor="rgba(142,202,230,0.12)",
                hovertemplate="Alt: %{y:.0f} m<extra></extra>",
            ))
            # Hidden axis scaled so the profile fills roughly the lower 40% of the plot
            fig_hr_pace.update_layout(yaxis3=dict(
                overlaying="y", visible=False, range=[a_lo - a_span * 0.05, a_lo + a_span * 2.5],
            ))
        if not rec.empty:
            fig_hr_pace.add_trace(go.Scatter(
                x=rec["distance_km"], y=rec["heart_rate"],
                mode="lines", name="FC", line=dict(color=ACCENT2, width=1.5), yaxis="y1",
                hovertemplate="FC: %{y} bpm<extra></extra>",
            ))
        if not rec_pace.empty:
            fig_hr_pace.add_trace(go.Scatter(
                x=rec_pace["distance_km"], y=rec_pace["pace_min_km"],
                mode="lines", name="Ritmo", line=dict(color=ACCENT, width=1.5), yaxis="y2",
                hovertemplate="Ritmo: %{customdata}/km<extra></extra>",
                customdata=[format_pace(p) for p in rec_pace["pace_min_km"]],
            ))
        for d in lap_marks:
            fig_hr_pace.add_vline(x=d, line_width=1, line_dash="dot",
                                  line_color="rgba(255,255,255,0.2)")
    fig_hr_pace.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=10, r=60, t=10, b=10), xaxis_title="Distancia (km)",
        yaxis=dict(title=dict(text="FC (bpm)", font=dict(color=ACCENT2)), tickfont=dict(color=ACCENT2)),
        yaxis2=dict(title=dict(text="Ritmo (min/km)", font=dict(color=ACCENT)), tickfont=dict(color=ACCENT),
                    overlaying="y", side="right", autorange="reversed",
                    tickvals=[4, 5, 6, 7, 8], ticktext=["4:00", "5:00", "6:00", "7:00", "8:00"]),
        legend=dict(orientation="h", y=1.1),
    )

    # Standalone HR chart with the activity average
    fig_hr = go.Figure()
    if not rec.empty:
        fig_hr.add_trace(go.Scatter(
            x=rec["distance_km"], y=rec["heart_rate"],
            mode="lines", line=dict(color=ACCENT2, width=1.5),
            fill="tozeroy", fillcolor="rgba(233,69,96,0.10)",
            hovertemplate="FC: %{y} bpm<extra></extra>",
        ))
        avg_hr = float(rec["heart_rate"].mean())
        fig_hr.add_hline(y=avg_hr, line_width=1, line_dash="dash", line_color="rgba(233,69,96,0.6)",
                         annotation_text=f"Media {avg_hr:.0f} bpm", annotation_position="top right",
                         annotation_font=dict(color=ACCENT2, size=10))
        for d in lap_marks:
            fig_hr.add_vline(x=d, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.2)")
        hr_lo, hr_hi = float(rec["heart_rate"].min()), float(rec["heart_rate"].max())
        fig_hr.update_yaxes(range=[hr_lo - 5, hr_hi + 5])
    fig_hr.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Distancia (km)", yaxis_title="FC (bpm)",
    )

    # Standalone pace chart (y reversed: faster is higher) with the activity average
    fig_pace = go.Figure()
    if not rec_pace.empty:
        fig_pace.add_trace(go.Scatter(
            x=rec_pace["distance_km"], y=rec_pace["pace_min_km"],
            mode="lines", line=dict(color=ACCENT, width=1.5),
            hovertemplate="Ritmo: %{customdata}/km<extra></extra>",
            customdata=[format_pace(p) for p in rec_pace["pace_min_km"]],
        ))
        avg_p = row["avg_pace_min_km"]
        if pd.notna(avg_p):
            fig_pace.add_hline(y=float(avg_p), line_width=1, line_dash="dash",
                               line_color="rgba(82,183,136,0.7)",
                               annotation_text=f"Media {format_pace(avg_p)}/km", annotation_position="top right",
                               annotation_font=dict(color=ACCENT, size=10))
        for d in lap_marks:
            fig_pace.add_vline(x=d, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.2)")
    fig_pace.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Distancia (km)",
        yaxis=dict(title="Ritmo (min/km)", autorange="reversed", **pace_ticks),
    )

    fig_alt = go.Figure()
    if not act_records.empty:
        alt_rec = act_records.dropna(subset=["altitude_m"])
        if not alt_rec.empty:
            fig_alt.add_trace(go.Scatter(
                x=alt_rec["distance_km"], y=alt_rec["altitude_m"],
                mode="lines", fill="tozeroy",
                line=dict(color="#8ECAE6", width=1.5), fillcolor="rgba(142,202,230,0.15)",
                hovertemplate="Alt: %{y:.0f} m<extra></extra>",
            ))
    fig_alt.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Distancia (km)", yaxis_title="Altitud (m)",
    )

    # Cumulative elevation gain per km
    fig_cum_elev = go.Figure()
    if not act_records.empty:
        alt_rec = act_records.dropna(subset=["altitude_m", "distance_km"]).sort_values("distance_km")
        if len(alt_rec) >= 2:
            smooth = alt_rec["altitude_m"].reset_index(drop=True).rolling(10, min_periods=1).mean()
            diffs = smooth.diff().clip(lower=0)
            cum_gain = diffs.cumsum()
            dist_vals = alt_rec["distance_km"].reset_index(drop=True)
            total_gain = float(cum_gain.iloc[-1])
            fig_cum_elev.add_trace(go.Scatter(
                x=dist_vals, y=cum_gain,
                mode="lines", fill="tozeroy",
                line=dict(color="#8ECAE6", width=2),
                fillcolor="rgba(142,202,230,0.15)",
                hovertemplate="D+ acumulado: %{y:.0f} m<extra></extra>",
            ))
            if not act_laps.empty:
                for d in act_laps["distance_km"].cumsum().iloc[:-1]:
                    fig_cum_elev.add_vline(x=d, line_width=1, line_dash="dot",
                                           line_color="rgba(255,255,255,0.2)")
            fig_cum_elev.add_annotation(
                x=float(dist_vals.iloc[-1]), y=total_gain,
                text=f"+{total_gain:.0f} m",
                showarrow=False, font=dict(color="#8ECAE6", size=12),
                xanchor="right", yanchor="bottom",
            )
    fig_cum_elev.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Distancia (km)", yaxis_title="D+ acumulado (m)",
    )

    fig_pc = go.Figure()
    if not act_records.empty:
        pow_rec = act_records.dropna(subset=["power_w"])
        pow_rec = pow_rec[pow_rec["power_w"] > 0]
        cad_rec = act_records.dropna(subset=["cadence_spm"])
        cad_rec = cad_rec[cad_rec["cadence_spm"] > 0]
        if not pow_rec.empty:
            fig_pc.add_trace(go.Scatter(
                x=pow_rec["distance_km"], y=pow_rec["power_w"],
                mode="lines", name="Potencia (W)", line=dict(color=ACCENT, width=1.5), yaxis="y1",
                hovertemplate="Potencia: %{y} W<extra></extra>",
            ))
        if not cad_rec.empty:
            # Centered rolling mean (~20 s at 1 Hz) to smooth step-to-step noise
            cad_rec = cad_rec.sort_values("distance_km")
            cad_smooth = cad_rec["cadence_spm"].rolling(20, center=True, min_periods=1).mean()
            fig_pc.add_trace(go.Scatter(
                x=cad_rec["distance_km"], y=cad_smooth,
                mode="lines", name="Cadencia (spm)", line=dict(color="#F4D03F", width=1.5), yaxis="y2",
                hovertemplate="Cadencia: %{y:.0f} spm<extra></extra>",
            ))
    fig_pc.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        margin=dict(l=10, r=60, t=10, b=10), xaxis_title="Distancia (km)",
        yaxis=dict(title=dict(text="Potencia (W)", font=dict(color=ACCENT)), tickfont=dict(color=ACCENT)),
        yaxis2=dict(title=dict(text="Cadencia (spm)", font=dict(color="#F4D03F")), tickfont=dict(color="#F4D03F"),
                    overlaying="y", side="right", tickformat=".0f"),
        legend=dict(orientation="h", y=1.1),
    )

    # GPS route map coloured by pace
    fig_map = go.Figure()
    gps = act_records.dropna(subset=["lat", "lon"])
    if not gps.empty:
        # Faint background line for the full route
        fig_map.add_trace(go.Scattermap(
            lat=gps["lat"], lon=gps["lon"],
            mode="lines",
            line=dict(width=3, color="rgba(255,255,255,0.12)"),
            hoverinfo="skip", showlegend=False,
        ))
        # Pace-coloured markers (sampled to ~600 points)
        gps_p = gps.dropna(subset=["pace_min_km"])
        gps_p = gps_p[gps_p["pace_min_km"].between(3.5, 10)].copy()
        if not gps_p.empty:
            step = max(1, len(gps_p) // 600)
            sample = gps_p.iloc[::step]
            p_lo = float(sample["pace_min_km"].quantile(0.05))
            p_hi = float(sample["pace_min_km"].quantile(0.95))
            tick_paces = [p for p in [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5] if p_lo <= p <= p_hi]
            if len(tick_paces) < 2:
                tick_paces = [p_lo, (p_lo + p_hi) / 2, p_hi]
            fig_map.add_trace(go.Scattermap(
                lat=sample["lat"], lon=sample["lon"],
                mode="markers",
                marker=dict(
                    size=6,
                    color=sample["pace_min_km"],
                    colorscale="RdYlGn_r",
                    cmin=p_lo, cmax=p_hi,
                    colorbar=dict(
                        title=dict(text="Ritmo<br>min/km", font=dict(color=TEXT, size=10)),
                        tickfont=dict(color=TEXT, size=9),
                        tickvals=tick_paces,
                        ticktext=[format_pace(p) for p in tick_paces],
                        thickness=12, len=0.5,
                    ),
                    opacity=0.9,
                ),
                hovertemplate="Km %{customdata[1]:.2f}<br>Ritmo: %{customdata[0]}/km<extra></extra>",
                # [pace label, distance km] — distance drives the synced cursor on the charts
                customdata=list(zip([format_pace(p) for p in sample["pace_min_km"]],
                                    sample["distance_km"].round(3))),
                showlegend=False,
            ))
        # Start / finish markers
        fig_map.add_trace(go.Scattermap(
            lat=[float(gps["lat"].iloc[0])], lon=[float(gps["lon"].iloc[0])],
            mode="markers", marker=dict(size=14, color=ACCENT),
            text=["Inicio"], hoverinfo="text", showlegend=False,
        ))
        fig_map.add_trace(go.Scattermap(
            lat=[float(gps["lat"].iloc[-1])], lon=[float(gps["lon"].iloc[-1])],
            mode="markers", marker=dict(size=14, color=ACCENT2),
            text=["Fin"], hoverinfo="text", showlegend=False,
        ))
        # Hover cursor (dark ring + white dot), moved client-side via Plotly.restyle
        for size, color in [(24, "#ffffff"), (18, PAGE_BG), (11, "#ffffff")]:
            fig_map.add_trace(go.Scattermap(
                lat=[], lon=[], mode="markers", name="km-cursor",
                marker=dict(size=size, color=color),
                hoverinfo="skip", showlegend=False,
            ))
        center_lat = float(gps["lat"].mean())
        center_lon = float(gps["lon"].mean())
        max_range = max(float(gps["lat"].max() - gps["lat"].min()),
                        float(gps["lon"].max() - gps["lon"].min()))
        zoom = max(10, min(15, round(13 - np.log2(max(max_range, 0.001) / 0.05))))
        fig_map.update_layout(
            map_style="carto-darkmatter",
            map=dict(center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
            paper_bgcolor=CARD_BG,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
        )
    else:
        fig_map = empty_fig("Sin datos GPS")

    track = None
    gps_d = gps.dropna(subset=["distance_km"]).sort_values("distance_km")
    if not gps_d.empty:
        gps_d = gps_d.iloc[::max(1, len(gps_d) // 3000)]
        track = {
            "dist": gps_d["distance_km"].round(4).tolist(),
            "lat": gps_d["lat"].round(6).tolist(),
            "lon": gps_d["lon"].round(6).tolist(),
        }

    # Unified hover so the cursor follows the mouse anywhere along the x axis
    for f in (fig_hr_pace, fig_hr, fig_pace, fig_alt, fig_cum_elev, fig_pc):
        f.update_layout(hovermode="x unified")
        f.update_xaxes(unifiedhovertitle=dict(text="Km %{x:.2f}"))

    return kpis, laps_table, fig_hr_pace, fig_hr, fig_pace, fig_alt, fig_cum_elev, fig_pc, fig_map, track


# Client-side: sync a vertical km line across the detail charts and a dot on the map
app.clientside_callback(
    dash.ClientsideFunction(namespace="detail", function_name="syncCursor"),
    Output("detail-cursor-dummy", "data"),
    Input("detail-hr-pace", "hoverData"),
    Input("detail-hr", "hoverData"),
    Input("detail-pace", "hoverData"),
    Input("detail-altitude", "hoverData"),
    Input("detail-cum-elevation", "hoverData"),
    Input("detail-power-cad", "hoverData"),
    Input("detail-map", "hoverData"),
    State("detail-track", "data"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
