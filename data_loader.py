"""Parses all .fit files in the same directory and returns structured DataFrames."""

import glob
import json
import os
import pickle
import re
import threading
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import fitparse
from pathlib import Path

# DATA_DIR overrides the data location (used by the Docker image, mounted at /data)
FIT_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent / "fit_raw"))
SEMICIRCLE_TO_DEG = 180.0 / (2**31)

HR_ZONE_LABELS = ["Z1 Recuperación", "Z2 Aeróbico", "Z3 Umbral aeróbico", "Z4 Umbral anaeróbico", "Z5 VO2 Max"]
HR_ZONE_COLORS = ["#8ECAE6", "#52B788", "#F4D03F", "#E67E22", "#E74C3C"]
REST_HR = 50  # resting heart rate (bpm) — used for Karvonen zone calculation
_KARVONEN_PCT = [0.60, 0.70, 0.80, 0.90]  # upper boundaries for Z1-Z4


def karvonen_boundaries(max_hr: float, rest_hr: float = REST_HR) -> list:
    """Return the 4 upper HR boundaries for zones Z1-Z4 using Karvonen method."""
    hrr = max_hr - rest_hr
    return [rest_hr + p * hrr for p in _KARVONEN_PCT]


def _compute_elevation(alt_series: pd.Series):
    """Return (gain_m, loss_m) from altitude series using 10-sample rolling mean."""
    if len(alt_series) < 2:
        return np.nan, np.nan
    smooth = alt_series.reset_index(drop=True).rolling(10, min_periods=1).mean()
    diffs = smooth.diff().dropna()
    gain = float(diffs[diffs > 0].sum())
    loss = float(abs(diffs[diffs < 0].sum()))
    return gain, loss


# Outlier limits applied while loading, so every chart and average sees clean data
PACE_MAX_MIN_KM = 12.0        # slower than this is standing/walking, not running pace
HR_VALID_RANGE = (30, 230)    # bpm; 0 means the sensor had no signal


def _speed_to_pace(speed_mps) -> float:
    """Convert speed in m/s to pace in min/km. Returns NaN when stopped or slower than PACE_MAX_MIN_KM."""
    if speed_mps and speed_mps > 0.3:
        pace = (1000.0 / speed_mps) / 60.0
        if pace <= PACE_MAX_MIN_KM:
            return pace
    return np.nan


def _clean_hr(hr):
    """Return the heart rate, or None when it is outside the physiologically valid range."""
    if hr is None or not (HR_VALID_RANGE[0] <= hr <= HR_VALID_RANGE[1]):
        return None
    return hr


def format_pace(pace_min_km) -> str:
    """Format pace as MM:SS/km string."""
    if pd.isna(pace_min_km) or pace_min_km > 20 or pace_min_km <= 0:
        return "—"
    mins = int(pace_min_km)
    secs = int(round((pace_min_km - mins) * 60))
    if secs == 60:
        mins += 1
        secs = 0
    return f"{mins}:{secs:02d}"


def format_duration(seconds) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    if pd.isna(seconds) or seconds is None:
        return "—"
    total_sec = int(seconds)
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def hr_zone(hr: float, max_hr: float, rest_hr: float = REST_HR) -> int:
    """Return 1-5 HR zone using Karvonen (Heart Rate Reserve) method."""
    if pd.isna(hr) or pd.isna(max_hr) or max_hr == 0:
        return 0
    for i, boundary in enumerate(karvonen_boundaries(max_hr, rest_hr)):
        if hr <= boundary:
            return i + 1
    return 5


def load_all_data():
    """
    Parse all .fit files and return (sessions_df, laps_df, records_df).

    sessions_df: one row per activity with aggregated stats.
    laps_df:     one row per km-lap with per-lap stats.
    records_df:  one row per GPS/sensor record (~1 Hz) for detailed charts.
    """
    sessions, laps, records = [], [], []
    fit_files = sorted(glob.glob(str(FIT_DIR / "*.fit")))

    for filepath in fit_files:
        name = Path(filepath).stem
        try:
            fit = fitparse.FitFile(filepath)

            for msg in fit.get_messages("session"):
                f = {fld.name: fld.value for fld in msg.fields if fld.value is not None}
                avg_spd = f.get("enhanced_avg_speed") or f.get("avg_speed") or 0
                max_spd = f.get("enhanced_max_speed") or f.get("max_speed") or 0
                sessions.append({
                    "name": name,
                    "start_time": f.get("start_time"),
                    "duration_sec": f.get("total_timer_time"),
                    "distance_km": (f.get("total_distance") or 0) / 1000.0,
                    "avg_pace_min_km": _speed_to_pace(avg_spd),
                    "best_pace_min_km": _speed_to_pace(max_spd),
                    "avg_hr": _clean_hr(f.get("avg_heart_rate")),
                    "max_hr": _clean_hr(f.get("max_heart_rate")),
                    "total_calories": f.get("total_calories"),
                    "avg_cadence_spm": (f.get("avg_running_cadence") or 0) * 2 or None,
                    "avg_power_w": f.get("avg_power"),
                    "max_power_w": f.get("max_power"),
                    "avg_vertical_osc_mm": f.get("avg_vertical_oscillation"),
                    "avg_stance_ms": f.get("avg_stance_time"),
                    "num_laps": f.get("num_laps"),
                    "total_strides": f.get("total_strides"),
                })

            lap_num = 0
            for msg in fit.get_messages("lap"):
                f = {fld.name: fld.value for fld in msg.fields if fld.value is not None}
                avg_spd = f.get("enhanced_avg_speed") or f.get("avg_speed") or 0
                laps.append({
                    "activity": name,
                    "lap_num": lap_num + 1,
                    "start_time": f.get("start_time"),
                    "duration_sec": f.get("total_timer_time"),
                    "distance_km": (f.get("total_distance") or 0) / 1000.0,
                    "avg_pace_min_km": _speed_to_pace(avg_spd),
                    "avg_hr": _clean_hr(f.get("avg_heart_rate")),
                    "max_hr": _clean_hr(f.get("max_heart_rate")),
                    "calories": f.get("total_calories"),
                    "avg_cadence_spm": (f.get("avg_cadence") or 0) * 2 or None,
                    "avg_power_w": f.get("avg_power"),
                    "avg_vertical_osc_mm": f.get("avg_vertical_oscillation"),
                    "avg_stance_ms": f.get("avg_stance_time"),
                })
                lap_num += 1

            for msg in fit.get_messages("record"):
                f = {fld.name: fld.value for fld in msg.fields if fld.value is not None}
                spd = f.get("enhanced_speed") or f.get("speed") or 0
                lat = f.get("position_lat")
                lon = f.get("position_long")
                records.append({
                    "activity": name,
                    "timestamp": f.get("timestamp"),
                    "distance_km": (f.get("distance") or 0) / 1000.0,
                    "speed_mps": spd,
                    "pace_min_km": _speed_to_pace(spd),
                    "heart_rate": _clean_hr(f.get("heart_rate")),
                    "cadence_spm": (f.get("cadence") or 0) * 2 or None,
                    "altitude_m": f.get("enhanced_altitude") or f.get("altitude"),
                    "power_w": f.get("power"),
                    "vertical_osc_mm": f.get("vertical_oscillation"),
                    "stance_ms": f.get("stance_time"),
                    "lat": lat * SEMICIRCLE_TO_DEG if lat else None,
                    "lon": lon * SEMICIRCLE_TO_DEG if lon else None,
                })

        except Exception as e:
            print(f"[WARNING] Could not load '{name}': {e}")

    sessions_df = pd.DataFrame(sessions)
    laps_df = pd.DataFrame(laps)
    records_df = pd.DataFrame(records)

    if not sessions_df.empty:
        sessions_df["start_time"] = pd.to_datetime(sessions_df["start_time"])
        sessions_df = sessions_df.sort_values("start_time").reset_index(drop=True)
        sessions_df["activity_num"] = range(1, len(sessions_df) + 1)
        sessions_df["date"] = sessions_df["start_time"].dt.date
        sessions_df["week"] = sessions_df["start_time"].dt.isocalendar().week
        sessions_df["week_start"] = sessions_df["start_time"].dt.to_period("W").dt.start_time
        sessions_df["month_label"] = sessions_df["start_time"].dt.strftime("%b %Y")
        sessions_df["duration_min"] = sessions_df["duration_sec"] / 60.0

    if not laps_df.empty:
        laps_df["start_time"] = pd.to_datetime(laps_df["start_time"])
        laps_df["duration_min"] = laps_df["duration_sec"] / 60.0

    if not records_df.empty:
        records_df["timestamp"] = pd.to_datetime(records_df["timestamp"])

    # Compute elevation gain/loss from altitude records (WorkOutDoors omits these in session msg)
    if not records_df.empty and not sessions_df.empty:
        elev_rows = []
        for act_name in sessions_df["name"]:
            alt = records_df.loc[
                (records_df["activity"] == act_name) & records_df["altitude_m"].notna(),
                "altitude_m",
            ]
            gain, loss = _compute_elevation(alt)
            elev_rows.append({"name": act_name, "total_ascent_m": gain, "total_descent_m": loss})
        sessions_df = sessions_df.merge(pd.DataFrame(elev_rows), on="name", how="left")

    if not records_df.empty and not laps_df.empty:
        lap_elev_rows = []
        for act_name in laps_df["activity"].unique():
            act_rec = records_df.loc[
                (records_df["activity"] == act_name)
                & records_df["altitude_m"].notna()
                & records_df["distance_km"].notna(),
            ].sort_values("distance_km")
            act_laps = laps_df[laps_df["activity"] == act_name].sort_values("lap_num")
            cum = 0.0
            for _, lap in act_laps.iterrows():
                lap_end = cum + lap["distance_km"]
                lap_alts = act_rec.loc[
                    (act_rec["distance_km"] >= cum) & (act_rec["distance_km"] < lap_end),
                    "altitude_m",
                ]
                gain, loss = _compute_elevation(lap_alts)
                lap_elev_rows.append({
                    "activity": act_name,
                    "lap_num": lap["lap_num"],
                    "total_ascent_m": gain,
                    "total_descent_m": loss,
                })
                cum = lap_end
        lap_elev_df = pd.DataFrame(lap_elev_rows)
        if not lap_elev_df.empty:
            laps_df = laps_df.merge(lap_elev_df, on=["activity", "lap_num"], how="left")

    return sessions_df, laps_df, records_df


def compute_hr_zone_distribution(records_df: pd.DataFrame, max_hr: float, rest_hr: float = REST_HR) -> pd.DataFrame:
    """Calculate minutes spent in each HR zone across all activities."""
    if records_df.empty or max_hr == 0:
        return pd.DataFrame({"zone": HR_ZONE_LABELS, "minutes": [0] * 5})

    df = records_df.dropna(subset=["heart_rate"]).copy()
    df["zone_num"] = df["heart_rate"].apply(lambda hr: hr_zone(hr, max_hr, rest_hr))
    zone_counts = df.groupby("zone_num").size().reset_index(name="records")
    # ~1 record per second → convert to minutes
    zone_counts["minutes"] = zone_counts["records"] / 60.0

    result = pd.DataFrame({"zone_num": range(1, 6), "zone": HR_ZONE_LABELS})
    result = result.merge(zone_counts, on="zone_num", how="left").fillna(0)
    result["minutes"] = result["minutes"].astype(float)
    return result


def weekly_summary(sessions_df: pd.DataFrame) -> pd.DataFrame:
    """Return weekly distance, time and activity count."""
    if sessions_df.empty:
        return pd.DataFrame()
    return (
        sessions_df.groupby("week_start")
        .agg(
            distance_km=("distance_km", "sum"),
            duration_min=("duration_min", "sum"),
            activities=("name", "count"),
            avg_hr=("avg_hr", "mean"),
            total_ascent_m=("total_ascent_m", "sum"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Session tags (user-defined labels such as "Series" or "Trail")
# ---------------------------------------------------------------------------
DEFAULT_TAGS = ["Entreno", "Series", "Carrera", "Trail", "Tirada larga", "Recuperación"]
UNTAGGED = "Sin etiqueta"  # pseudo-tag used by the filter for sessions without a label
MAX_TAG_LEN = 30

# Tags live inside the data folder (fit_raw/), so copying that folder to another machine
# carries the labels with it. TAGS_FILE overrides the location. The folder must be writable.
TAGS_FILE = Path(os.environ.get("TAGS_FILE", FIT_DIR / "activity_tags.json"))
_TAGS_LOCK = threading.Lock()


def _read_tags_file() -> dict:
    try:
        raw = json.loads(TAGS_FILE.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_tags_file(raw: dict) -> None:
    """Write atomically so a crash mid-write never corrupts the labels."""
    TAGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = TAGS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, TAGS_FILE)


def _normalize_tag_list(value) -> list:
    """Accept a list of tags or a legacy single string; return unique non-empty tags in order."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return list(dict.fromkeys(t for t in value if isinstance(t, str) and t))


def load_tag_state() -> dict:
    """Return {"options": [tag, ...], "assignments": {activity_name: [tag, ...]}}.

    Older files stored a single string per session; those are read as one-element lists.
    """
    raw = _read_tags_file()
    custom = [t for t in (raw.get("custom") or []) if isinstance(t, str) and t]
    assignments = {}
    for name, value in (raw.get("assignments") or {}).items():
        tags = _normalize_tag_list(value)
        if tags:
            assignments[name] = tags
    options = list(DEFAULT_TAGS)
    for tag in custom + [t for tags in assignments.values() for t in tags]:
        if tag not in options:
            options.append(tag)
    return {"options": options, "assignments": assignments}


def set_session_tags(changes: dict) -> None:
    """Apply {activity_name: [tags]}. An empty list/None removes every label from the session."""
    with _TAGS_LOCK:
        raw = _read_tags_file()
        assignments = dict(raw.get("assignments") or {})
        for name, tags in changes.items():
            tags = _normalize_tag_list(tags)
            if tags:
                assignments[name] = tags
            else:
                assignments.pop(name, None)
        raw["assignments"] = assignments
        _write_tags_file(raw)


def add_custom_tag(label: str):
    """Register a new tag. Returns (normalized_label, error); exactly one of them is None."""
    label = re.sub(r"\s+", " ", (label or "")).strip()
    if not label:
        return None, "Escribe un nombre para la etiqueta."
    if len(label) > MAX_TAG_LEN:
        return None, f"Máximo {MAX_TAG_LEN} caracteres."
    with _TAGS_LOCK:
        existing = load_tag_state()["options"] + [UNTAGGED]
        if label.lower() in (t.lower() for t in existing):
            return None, f"La etiqueta «{label}» ya existe."
        raw = _read_tags_file()
        raw["custom"] = list(raw.get("custom") or []) + [label]
        _write_tags_file(raw)
    return label, None


# ---------------------------------------------------------------------------
# Interval ("series") detection: separate work reps from recovery
# ---------------------------------------------------------------------------
SERIES_TAG = "Series"          # sessions carrying this tag are analysed for intervals
SMOOTH_SEC = 7                 # speed smoothing window (seconds)
MIN_REP_SEC = 15               # shorter fast bursts are ignored
MERGE_GAP_SEC = 5              # slow gaps shorter than this are merged into the surrounding rep
MIN_SPEED_RATIO = 1.3          # work speed must be at least this much faster than recovery
MIN_REPS = 2
MOVING_SPEED_MPS = 0.8         # below this the runner is considered stopped
MAX_RECORD_GAP_SEC = 5         # longer holes between records are treated as standing still
SHORT_BLOCK_FACTOR = 0.4       # blocks shorter than this share of the typical rep are noise
LONG_BLOCK_FACTOR = 2.2        # a first/last block this many times longer than the typical rep is warm-up/cool-down


def _otsu_threshold(values: np.ndarray) -> float:
    """Speed threshold that best splits the samples into a slow and a fast group (Otsu)."""
    candidates = np.linspace(values.min(), values.max(), 80)[1:-1]
    best_thr, best_var = np.nan, -1.0
    for thr in candidates:
        lo, hi = values[values <= thr], values[values > thr]
        if len(lo) == 0 or len(hi) == 0:
            continue
        w_lo, w_hi = len(lo) / len(values), len(hi) / len(values)
        var = w_lo * w_hi * (lo.mean() - hi.mean()) ** 2
        if var > best_var:
            best_thr, best_var = thr, var
    return best_thr


def detect_intervals(act_records: pd.DataFrame):
    """
    Find the fast "work" repetitions of one interval session.

    Works on a 1 Hz grid: speed is smoothed, split into slow/fast with an Otsu threshold and the
    fast runs become reps. Returns (reps_df, summary) or (None, None) when the session does not
    show two clearly different speeds or has fewer than MIN_REPS reps (e.g. a steady run).
    """
    d = act_records.dropna(subset=["timestamp", "distance_km"]).sort_values("timestamp")
    if len(d) < 60:
        return None, None
    secs = (d["timestamp"] - d["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
    secs, uniq = np.unique(secs, return_index=True)
    d = d.iloc[uniq]
    n = int(secs[-1]) + 1
    if n < 120:
        return None, None
    grid = np.arange(n, dtype=float)
    # Stopped / walking samples (no valid pace) count as speed 0
    speed = np.where(d["pace_min_km"].isna(), 0.0, d["speed_mps"].fillna(0.0)).astype(float)
    speed_g = np.interp(grid, secs, speed)
    # Recording gaps (watch auto-pause, lost signal) are time without movement, not interpolated speed
    right = np.clip(np.searchsorted(secs, grid), 1, len(secs) - 1)
    in_gap = (secs[right] - secs[right - 1]) > MAX_RECORD_GAP_SEC
    speed_g[in_gap & ~np.isin(grid, secs)] = 0.0
    dist_g = np.interp(grid, secs, d["distance_km"].to_numpy(float))
    hr = d["heart_rate"].to_numpy(float)
    ok = ~np.isnan(hr)
    hr_g = np.interp(grid, secs[ok], hr[ok]) if ok.sum() >= 2 else np.full(n, np.nan)

    smooth = pd.Series(speed_g).rolling(SMOOTH_SEC, center=True, min_periods=1).mean().to_numpy()
    # The threshold is learned from moving samples only: long stops must not decide where
    # "slow" ends (stopped time is always recovery).
    moving = smooth >= MOVING_SPEED_MPS
    if moving.sum() < 60 or smooth[moving].max() - smooth[moving].min() < 0.5:
        return None, None
    thr = _otsu_threshold(smooth[moving])
    if np.isnan(thr):
        return None, None
    fast = smooth > thr
    share_fast = fast.mean()
    if share_fast < 0.08 or share_fast > 0.92:
        return None, None
    slow_moving = moving & ~fast
    slow_mean = smooth[slow_moving].mean() if slow_moving.any() else MOVING_SPEED_MPS
    if smooth[fast].mean() / max(slow_mean, MOVING_SPEED_MPS) < MIN_SPEED_RATIO:
        return None, None

    # Runs of fast samples -> [start, end) in seconds
    edges = np.flatnonzero(np.diff(np.concatenate(([0], fast.astype(int), [0]))))
    runs = [[int(a), int(b)] for a, b in zip(edges[::2], edges[1::2])]
    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= MERGE_GAP_SEC:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    merged = [r for r in merged if r[1] - r[0] >= MIN_REP_SEC]
    # Tiny blips (GPS noise, a short surge) are much shorter than a real rep
    if len(merged) >= MIN_REPS:
        typical = float(np.median([r[1] - r[0] for r in merged]))
        merged = [r for r in merged if r[1] - r[0] >= SHORT_BLOCK_FACTOR * typical]
    # A steady warm-up / cool-down is faster than the threshold too: drop it when it is much
    # longer than the typical rep
    for edge in (0, -1):
        if len(merged) > MIN_REPS:
            others = [r[1] - r[0] for i, r in enumerate(merged) if i != (edge % len(merged))]
            if merged[edge][1] - merged[edge][0] > LONG_BLOCK_FACTOR * float(np.median(others)):
                merged.pop(edge)
    if len(merged) < MIN_REPS:
        return None, None

    rows, prev_end = [], None
    for i, (a, b) in enumerate(merged, 1):
        b = min(b, n - 1)
        dur, km = b - a, dist_g[b] - dist_g[a]
        if dur <= 0 or km <= 0:
            continue
        rows.append({
            "rep": i, "start_sec": a, "end_sec": b, "start_km": dist_g[a], "end_km": dist_g[b],
            "duration_sec": dur, "distance_km": km, "pace_min_km": dur / 60.0 / km,
            "avg_hr": float(np.nanmean(hr_g[a:b])) if not np.isnan(hr_g[a:b]).all() else np.nan,
            "rest_before_sec": (a - prev_end) if prev_end is not None else np.nan,
        })
        prev_end = b
    if len(rows) < MIN_REPS:
        return None, None
    reps = pd.DataFrame(rows)
    reps["rep"] = range(1, len(reps) + 1)

    work_sec, work_km = reps["duration_sec"].sum(), reps["distance_km"].sum()
    hr_w = (reps["avg_hr"] * reps["duration_sec"]).sum() / reps.loc[reps["avg_hr"].notna(), "duration_sec"].sum() \
        if reps["avg_hr"].notna().any() else np.nan
    summary = {
        "n_reps": len(reps),
        "work_pace_min_km": work_sec / 60.0 / work_km,
        "work_hr": hr_w,
        "avg_rep_sec": reps["duration_sec"].mean(),
        "avg_rep_km": reps["distance_km"].mean(),
        "rest_sec": float(reps["rest_before_sec"].sum()),
        "avg_rest_sec": float(reps["rest_before_sec"].mean()),
        "threshold_pace_min_km": 1000.0 / thr / 60.0 if thr > 0 else np.nan,
    }
    return reps, summary


_APPLE_HEALTH_XML = FIT_DIR / "apple_health_export" / "exportación.xml"
# CACHE_DIR lets the cache live outside a read-only data mount (e.g. in Docker)
_APPLE_HEALTH_CACHE = Path(os.environ.get("CACHE_DIR", FIT_DIR / "apple_health_export")) / ".health_cache.pkl"

_AH_TARGETS = {
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
    "HKQuantityTypeIdentifierRestingHeartRate",
    "HKQuantityTypeIdentifierVO2Max",
}


def load_apple_health_data():
    """
    Parse Apple Health XML and return (hrv_df, resting_hr_df, vo2max_df).

    Results are cached as a pickle; the cache is invalidated when the XML changes.
    """
    if not _APPLE_HEALTH_XML.exists():
        empty = pd.DataFrame()
        return empty, empty, empty

    xml_mtime = _APPLE_HEALTH_XML.stat().st_mtime
    if _APPLE_HEALTH_CACHE.exists():
        try:
            with open(_APPLE_HEALTH_CACHE, "rb") as f:
                cache = pickle.load(f)
            if cache.get("mtime") == xml_mtime:
                return cache["hrv"], cache["rhr"], cache["vo2max"]
        except Exception:
            pass

    raw: dict[str, list] = {t: [] for t in _AH_TARGETS}
    for _event, elem in ET.iterparse(str(_APPLE_HEALTH_XML), events=("end",)):
        if elem.tag == "Record":
            rtype = elem.get("type")
            if rtype in _AH_TARGETS:
                try:
                    raw[rtype].append({
                        "date": pd.to_datetime(elem.get("startDate")),
                        "value": float(elem.get("value", "nan")),
                    })
                except (ValueError, TypeError):
                    pass
        elem.clear()

    def _to_df(rows):
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date", "value"])

    hrv_df = _to_df(raw["HKQuantityTypeIdentifierHeartRateVariabilitySDNN"])
    rhr_df = _to_df(raw["HKQuantityTypeIdentifierRestingHeartRate"])
    vo2_df = _to_df(raw["HKQuantityTypeIdentifierVO2Max"])

    if not hrv_df.empty:
        hrv_df["day"] = hrv_df["date"].dt.date
        hrv_df = (
            hrv_df.groupby("day")["value"]
            .agg(hrv_ms="median", hrv_lo="min", hrv_hi="max")
            .reset_index()
        )
        hrv_df["day"] = pd.to_datetime(hrv_df["day"])
        hrv_df = hrv_df.sort_values("day").reset_index(drop=True)
        hrv_df["hrv_7d"] = hrv_df["hrv_ms"].rolling(7, min_periods=1).mean()

    if not rhr_df.empty:
        rhr_df["day"] = pd.to_datetime(rhr_df["date"].dt.date)
        rhr_df = rhr_df.rename(columns={"value": "resting_hr"}).drop(columns=["date"])
        rhr_df = rhr_df.sort_values("day").reset_index(drop=True)
        rhr_df["rhr_7d"] = rhr_df["resting_hr"].rolling(7, min_periods=1).mean()

    if not vo2_df.empty:
        vo2_df["day"] = pd.to_datetime(vo2_df["date"].dt.date)
        vo2_df = vo2_df.rename(columns={"value": "vo2max"}).drop(columns=["date"])
        vo2_df = vo2_df.sort_values("day").reset_index(drop=True)

    try:
        _APPLE_HEALTH_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(_APPLE_HEALTH_CACHE, "wb") as f:
            pickle.dump({"mtime": xml_mtime, "hrv": hrv_df, "rhr": rhr_df, "vo2max": vo2_df}, f)
    except Exception:
        pass

    return hrv_df, rhr_df, vo2_df
