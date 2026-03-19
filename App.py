from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Optional pm4py support
# -----------------------------------------------------------------------------
try:
    import pm4py
    from pm4py import discover_petri_net_inductive
    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
    from pm4py.objects.conversion.log import converter as log_converter
    from pm4py.visualization.petri_net import visualizer as pn_visualizer

    PM4PY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    PM4PY_AVAILABLE = False


# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------
APP_TITLE = "🔎 Process Mining Explorer"
APP_DESCRIPTION = (
    "Upload a CSV event log, map the required columns, and explore the process with "
    "directly-follows graphs, Petri net discovery, conformance checking, and analytics."
)

CASE_SYNONYMS = [
    "case_id",
    "case id",
    "caseid",
    "case",
    "contract_id",
    "contract id",
    "id",
]
ACTIVITY_SYNONYMS = ["activity", "task", "event", "step", "action", "status"]
TIMESTAMP_SYNONYMS = [
    "timestamp",
    "time:timestamp",
    "time",
    "datetime",
    "date",
    "event_time",
    "event time",
    "created_at",
]

DEFAULT_TOP_N = 20
MAX_DFG_NODES = 50
MAX_DFG_EDGES = 200

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

st.set_page_config(page_title="Process Mining Explorer", layout="wide")


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class AppConfig:
    model_source: str
    miner_label: str
    heuristics_threshold: float
    show_full_dfg: bool
    show_section_dfgs: bool
    show_petri_net: bool
    conformance_metric: str
    timestamp_mode: str
    timestamp_format: Optional[str]


@dataclass(frozen=True)
class ColumnMapping:
    case_id: str
    activity: str
    timestamp: str


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def _safe_text(value: object) -> str:
    return str(value).replace('"', '\\"')


def _human_duration(seconds: Optional[float]) -> str:
    if seconds is None or pd.isna(seconds):
        return "n/a"
    if seconds < 0:
        return "n/a"
    return str(timedelta(seconds=int(seconds)))


def _percentage(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else (numerator / denominator) * 100.0


def _default_index(columns: List[str], candidates: Iterable[str], fallback: int) -> int:
    lower_map = {str(col).strip().lower(): idx for idx, col in enumerate(columns)}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    if not columns:
        return 0
    return max(0, min(fallback, len(columns) - 1))


def _df_download(name: str, df: pd.DataFrame, label: str) -> None:
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=name,
        mime="text/csv",
    )


# -----------------------------------------------------------------------------
# Cached data loading and transformations
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def prepare_event_log(
    df: pd.DataFrame,
    case_col: str,
    activity_col: str,
    timestamp_col: str,
    timestamp_mode: str,
    timestamp_format: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Parse, validate, clean, and sort the event log.

    Returns:
        cleaned_df, diagnostics
    """
    work = df.copy()

    required_cols = [case_col, activity_col, timestamp_col]
    missing = [col for col in required_cols if col not in work.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    original_rows = len(work)

    work[case_col] = work[case_col].astype(str).str.strip()
    work[activity_col] = work[activity_col].astype(str).str.strip()

    empty_required = (
        work[case_col].eq("")
        | work[activity_col].eq("")
        | work[case_col].isna()
        | work[activity_col].isna()
    )
    removed_empty_required = int(empty_required.sum())
    work = work.loc[~empty_required].copy()

    if timestamp_mode == "Provide a format":
        work[timestamp_col] = pd.to_datetime(
            work[timestamp_col],
            format=timestamp_format,
            errors="coerce",
        )
    else:
        work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")

    removed_bad_timestamps = int(work[timestamp_col].isna().sum())
    work = work.loc[work[timestamp_col].notna()].copy()

    work.sort_values([case_col, timestamp_col, activity_col], inplace=True)
    work.reset_index(drop=True, inplace=True)

    diagnostics = {
        "original_rows": int(original_rows),
        "remaining_rows": int(len(work)),
        "removed_empty_required": removed_empty_required,
        "removed_bad_timestamps": removed_bad_timestamps,
    }
    return work, diagnostics


@st.cache_data(show_spinner=False)
def build_follows_table(
    df: pd.DataFrame,
    case_col: str,
    activity_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    """Create a row-per-event table with next activity and next timestamp per case."""
    tmp = df[[case_col, activity_col, timestamp_col]].copy()
    tmp["next_activity"] = tmp.groupby(case_col)[activity_col].shift(-1)
    tmp["next_timestamp"] = tmp.groupby(case_col)[timestamp_col].shift(-1)
    tmp["delta_seconds"] = (tmp["next_timestamp"] - tmp[timestamp_col]).dt.total_seconds()
    return tmp


@st.cache_data(show_spinner=False)
def build_dfg_summary(
    df: pd.DataFrame,
    case_col: str,
    activity_col: str,
    timestamp_col: str,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int], Dict[str, int]]:
    follows = build_follows_table(df, case_col, activity_col, timestamp_col)
    transitions = follows.loc[follows["next_activity"].notna()].copy()

    dfg = (
        transitions.groupby([activity_col, "next_activity"], dropna=False)
        .agg(
            frequency=(activity_col, "size"),
            avg_duration_seconds=("delta_seconds", "mean"),
            median_duration_seconds=("delta_seconds", "median"),
        )
        .reset_index()
        .rename(columns={activity_col: "from_activity", "next_activity": "to_activity"})
        .sort_values(["frequency", "from_activity", "to_activity"], ascending=[False, True, True])
        .reset_index(drop=True)
    )

    start_activities = (
        df.groupby(case_col, sort=False)[activity_col]
        .first()
        .value_counts()
        .to_dict()
    )
    end_activities = (
        df.groupby(case_col, sort=False)[activity_col]
        .last()
        .value_counts()
        .to_dict()
    )
    activity_counts = df[activity_col].value_counts().to_dict()

    return dfg, activity_counts, start_activities, end_activities


@st.cache_data(show_spinner=False)
def performance_dfg_dot(
    df: pd.DataFrame,
    case_col: str,
    activity_col: str,
    timestamp_col: str,
    max_nodes: int = MAX_DFG_NODES,
    max_edges: int = MAX_DFG_EDGES,
) -> str:
    dfg_df, activity_counts, _, _ = build_dfg_summary(df, case_col, activity_col, timestamp_col)

    top_nodes = dict(sorted(activity_counts.items(), key=lambda item: item[1], reverse=True)[:max_nodes])
    filtered_edges = dfg_df[
        dfg_df["from_activity"].isin(top_nodes)
        & dfg_df["to_activity"].isin(top_nodes)
    ].head(max_edges)

    lines = [
        "digraph G {",
        "  rankdir=LR;",
        '  node [shape=box, style="rounded,filled", color="#4B5563", fillcolor="#E5E7EB"];',
    ]

    for activity, count in top_nodes.items():
        label = f"{_safe_text(activity)}\\n({count} events)"
        lines.append(f'  "{_safe_text(activity)}" [label="{label}"];')

    for row in filtered_edges.itertuples(index=False):
        duration_label = _human_duration(row.avg_duration_seconds)
        edge_label = f"{int(row.frequency)} | {duration_label}"
        lines.append(
            f'  "{_safe_text(row.from_activity)}" -> "{_safe_text(row.to_activity)}" '
            f'[label="{edge_label}", color="#2563EB"];'
        )

    lines.append("}")
    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def compute_log_summary(df: pd.DataFrame, case_col: str, timestamp_col: str) -> Dict[str, object]:
    if df.empty:
        return {
            "events": 0,
            "cases": 0,
            "first_event": None,
            "last_event": None,
            "avg_events_per_case": 0.0,
        }

    case_sizes = df.groupby(case_col).size()
    return {
        "events": int(len(df)),
        "cases": int(df[case_col].nunique()),
        "first_event": df[timestamp_col].min(),
        "last_event": df[timestamp_col].max(),
        "avg_events_per_case": float(case_sizes.mean()),
    }


@st.cache_data(show_spinner=False)
def compute_section_summary(df: pd.DataFrame, case_col: str) -> Optional[Dict[str, object]]:
    if "Section" not in df.columns:
        return None

    section_sets = (
        df.groupby(case_col, dropna=False)["Section"]
        .apply(lambda series: set(series.dropna().unique().tolist()))
        .reset_index(name="sections_used")
    )

    only_section_1 = section_sets.loc[section_sets["sections_used"] == {1}, case_col].tolist()
    only_section_2 = section_sets.loc[section_sets["sections_used"] == {2}, case_col].tolist()
    both_sections = section_sets.loc[
        section_sets["sections_used"].apply(lambda value: value == {1, 2}), case_col
    ].tolist()

    common_cases = (
        df.groupby(case_col)["Section"].nunique().loc[lambda s: s >= 2].index.tolist()
    )

    return {
        "only_section_1": only_section_1,
        "only_section_2": only_section_2,
        "both_sections": both_sections,
        "common_cases": common_cases,
    }


@st.cache_data(show_spinner=False)
def compute_case_summary(
    df: pd.DataFrame,
    case_col: str,
    activity_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    case_lengths = df.groupby(case_col)[activity_col].size().rename("events_per_case")
    case_times = df.groupby(case_col).agg(start=(timestamp_col, "min"), end=(timestamp_col, "max"))
    case_times["throughput_seconds"] = (case_times["end"] - case_times["start"]).dt.total_seconds()

    summary = case_lengths.to_frame().join(case_times, how="left").reset_index()
    summary.rename(columns={case_col: "case_id"}, inplace=True)
    return summary


@st.cache_data(show_spinner=False)
def compute_variants(df: pd.DataFrame, case_col: str, activity_col: str) -> pd.DataFrame:
    variants = (
        df.groupby(case_col)[activity_col]
        .apply(tuple)
        .value_counts()
        .reset_index()
    )
    variants.columns = ["variant", "number_of_cases"]
    variants["cumulative_cases"] = variants["number_of_cases"].cumsum()
    variants["cumulative_pct"] = variants["cumulative_cases"] / variants["number_of_cases"].sum() * 100.0
    variants["variant_display"] = variants["variant"].astype(str).str.slice(0, 120)
    return variants


@st.cache_data(show_spinner=False)
def compute_activity_summary(
    df: pd.DataFrame,
    case_col: str,
    activity_col: str,
    timestamp_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    activity_freq = df[activity_col].value_counts().reset_index()
    activity_freq.columns = ["Activity", "Events"]

    follows = build_follows_table(df, case_col, activity_col, timestamp_col)
    service = (
        follows.groupby(activity_col)["delta_seconds"]
        .agg(count="count", mean="mean", median="median")
        .reset_index()
        .rename(
            columns={
                activity_col: "Activity",
                "count": "Transitions observed",
                "mean": "Average time to next activity (days)",
                "median": "Median time to next activity (days)",
            }
        )
    )
    service["Average time to next activity (days)"] = service["Average time to next activity (days)"] / (24 * 3600)
    service["Median time to next activity (days)"] = service["Median time to next activity (days)"] / (24 * 3600)
    return activity_freq, service


@st.cache_data(show_spinner=False)
def compute_transition_pairs(
    df: pd.DataFrame,
    case_col: str,
    activity_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    follows = build_follows_table(df, case_col, activity_col, timestamp_col)
    transitions = follows.loc[follows["next_activity"].notna()].copy()
    transitions.rename(
        columns={
            case_col: "case_id",
            activity_col: "from_activity",
            "next_activity": "to_activity",
            "delta_seconds": "delta_seconds",
        },
        inplace=True,
    )
    return transitions[["case_id", "from_activity", "to_activity", "delta_seconds"]]


@st.cache_data(show_spinner=False)
def compute_transition_stats(pairs_df: pd.DataFrame) -> pd.DataFrame:
    if pairs_df.empty:
        return pd.DataFrame(columns=["from_activity", "to_activity", "count", "avg_days", "p50_days", "p90_days"])

    stats = (
        pairs_df.groupby(["from_activity", "to_activity"])["delta_seconds"]
        .agg(
            count="count",
            avg="mean",
            p50="median",
            p90=lambda x: float(np.percentile(x, 90)),
            p95=lambda x: float(np.percentile(x, 95)),
        )
        .reset_index()
    )
    for column in ["avg", "p50", "p90", "p95"]:
        stats[f"{column}_days"] = stats[column] / (24 * 3600)
    return stats.sort_values("avg", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def compute_wip_series(df: pd.DataFrame, case_col: str, timestamp_col: str) -> pd.DataFrame:
    case_times = df.groupby(case_col).agg(start=(timestamp_col, "min"), end=(timestamp_col, "max")).dropna()
    if case_times.empty:
        return pd.DataFrame(columns=["timestamp", "wip"])

    events = pd.concat(
        [
            case_times.reset_index()[["start"]].rename(columns={"start": "timestamp"}).assign(delta=1),
            case_times.reset_index()[["end"]].rename(columns={"end": "timestamp"}).assign(delta=-1),
        ],
        ignore_index=True,
    ).sort_values("timestamp")

    events["wip"] = events["delta"].cumsum()
    return events[["timestamp", "wip"]]


@st.cache_data(show_spinner=False)
def compute_skip_summary(
    df: pd.DataFrame,
    case_col: str,
    activity_col: str,
    activities_universe: Tuple[str, ...],
) -> pd.DataFrame:
    all_cases = set(df[case_col].dropna().astype(str).unique())
    total_cases = len(all_cases)
    cases_by_activity = (
        df.groupby(activity_col)[case_col]
        .apply(lambda values: set(values.dropna().astype(str).unique()))
        .to_dict()
    )

    rows = []
    for activity in activities_universe:
        present = cases_by_activity.get(activity, set())
        missing = sorted(all_cases - present)
        rows.append(
            {
                "Activity": activity,
                "Cases skipping": len(missing),
                "Share skipping (%)": _percentage(len(missing), total_cases),
                "Cases present": len(present),
                "Total cases": total_cases,
                "Missing case IDs": missing,
            }
        )

    return pd.DataFrame(rows).sort_values(["Cases skipping", "Activity"], ascending=[False, True]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# pm4py helpers
# -----------------------------------------------------------------------------
def to_pm4py_event_log(
    df: pd.DataFrame,
    case_col: str,
    activity_col: str,
    timestamp_col: str,
):
    if not PM4PY_AVAILABLE:
        raise RuntimeError("pm4py is not installed.")

    tmp = df[[case_col, activity_col, timestamp_col]].copy()
    tmp.rename(
        columns={
            case_col: "case:concept:name",
            activity_col: "concept:name",
            timestamp_col: "time:timestamp",
        },
        inplace=True,
    )
    tmp["time:timestamp"] = pd.to_datetime(tmp["time:timestamp"], errors="coerce")
    tmp = tmp.loc[tmp["time:timestamp"].notna()].copy()

    try:
        return log_converter.apply(tmp, variant=log_converter.Variants.TO_EVENT_LOG)
    except Exception:
        params = {}
        try:
            parameters = log_converter.Variants.TO_EVENT_LOG.value.Parameters
            if hasattr(parameters, "CASE_ID_KEY"):
                params[parameters.CASE_ID_KEY] = "case:concept:name"
            if hasattr(parameters, "ACTIVITY_KEY"):
                params[parameters.ACTIVITY_KEY] = "concept:name"
            if hasattr(parameters, "TIMESTAMP_KEY"):
                params[parameters.TIMESTAMP_KEY] = "time:timestamp"
        except Exception:
            params = {}
        return log_converter.apply(tmp, variant=log_converter.Variants.TO_EVENT_LOG, parameters=params)


def convert_dfg_to_petri(dfg_dict, start_acts, end_acts):
    if not PM4PY_AVAILABLE:
        raise RuntimeError("pm4py is not installed.")

    try:
        from pm4py.algo.discovery.dfg import algorithm as dfg_algo

        return dfg_algo.apply_dfg(dfg_dict, start_acts, end_acts)
    except Exception:
        pass

    try:
        from pm4py.objects.conversion.dfg import converter as dfg_converter

        return dfg_converter.apply(
            dfg_dict,
            parameters={"start_activities": start_acts, "end_activities": end_acts},
        )
    except Exception as exc:
        raise RuntimeError("No compatible DFG → Petri conversion API was found in pm4py.") from exc


def build_model_spec(
    model_name: str,
    dfg_dict: Dict[Tuple[str, str], int],
    start_acts: Dict[str, int],
    end_acts: Dict[str, int],
) -> Dict[str, object]:
    """Create a portable model specification that can be saved as JSON text."""
    transitions = [
        {"from": source, "to": target, "frequency": int(freq)}
        for (source, target), freq in sorted(dfg_dict.items(), key=lambda item: (item[0][0], item[0][1]))
    ]

    return {
        "format_version": 1,
        "model_name": model_name,
        "transitions": transitions,
        "start_activities": {str(activity): int(count) for activity, count in start_acts.items()},
        "end_activities": {str(activity): int(count) for activity, count in end_acts.items()},
    }


def export_model_spec_txt_bytes(model_spec: Dict[str, object]) -> bytes:
    """Serialize model specification to UTF-8 JSON text."""
    return json.dumps(model_spec, indent=2, ensure_ascii=False).encode("utf-8")


def load_model_spec_from_txt(file_bytes: bytes) -> Dict[str, object]:
    """Load and validate a model specification from UTF-8 JSON text."""
    try:
        spec = json.loads(file_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Could not parse model TXT file as JSON text: {exc}") from exc

    required_keys = {"transitions", "start_activities", "end_activities"}
    missing_keys = required_keys - set(spec.keys())
    if missing_keys:
        raise ValueError(f"Model TXT is missing required key(s): {', '.join(sorted(missing_keys))}")

    transitions = spec.get("transitions", [])
    if not isinstance(transitions, list):
        raise ValueError("'transitions' must be a list.")

    for index, transition in enumerate(transitions):
        if not isinstance(transition, dict):
            raise ValueError(f"Transition #{index + 1} must be an object.")
        if "from" not in transition or "to" not in transition:
            raise ValueError(f"Transition #{index + 1} must contain 'from' and 'to'.")
        if "frequency" in transition and int(transition["frequency"]) < 1:
            raise ValueError(f"Transition #{index + 1} has invalid frequency.")

    if not isinstance(spec.get("start_activities"), dict):
        raise ValueError("'start_activities' must be an object.")
    if not isinstance(spec.get("end_activities"), dict):
        raise ValueError("'end_activities' must be an object.")

    return spec


def model_spec_to_dfg_components(
    model_spec: Dict[str, object],
) -> Tuple[Dict[Tuple[str, str], int], Dict[str, int], Dict[str, int]]:
    """Convert a model specification into DFG/start/end dictionaries."""
    dfg_dict: Dict[Tuple[str, str], int] = {}
    for transition in model_spec["transitions"]:
        source = str(transition["from"]).strip()
        target = str(transition["to"]).strip()
        frequency = int(transition.get("frequency", 1))
        if not source or not target:
            raise ValueError("Transition activities may not be empty.")
        dfg_dict[(source, target)] = frequency

    start_acts = {str(activity).strip(): int(count) for activity, count in model_spec["start_activities"].items()}
    end_acts = {str(activity).strip(): int(count) for activity, count in model_spec["end_activities"].items()}

    start_acts = {activity: count for activity, count in start_acts.items() if activity}
    end_acts = {activity: count for activity, count in end_acts.items() if activity}

    if not dfg_dict:
        raise ValueError("Model TXT does not contain any transitions.")
    if not start_acts:
        raise ValueError("Model TXT does not contain any start activities.")
    if not end_acts:
        raise ValueError("Model TXT does not contain any end activities.")

    return dfg_dict, start_acts, end_acts


def build_petri_net_from_model_spec(model_spec: Dict[str, object]):
    """Build a Petri net from a saved model specification."""
    dfg_dict, start_acts, end_acts = model_spec_to_dfg_components(model_spec)
    net, initial_marking, final_marking = convert_dfg_to_petri(dfg_dict, start_acts, end_acts)
    return net, initial_marking, final_marking, dfg_dict, start_acts, end_acts


def discover_petri_model(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    config: AppConfig,
) -> Tuple[Optional[object], Optional[object], Optional[object], str]:
    if not PM4PY_AVAILABLE:
        return None, None, None, "pm4py not available"

    log = to_pm4py_event_log(df, mapping.case_id, mapping.activity, mapping.timestamp)

    if config.miner_label.startswith("Inductive"):
        net, initial_marking, final_marking = discover_petri_net_inductive(log)
        return net, initial_marking, final_marking, "Inductive Miner"

    if config.miner_label.startswith("Heuristics"):
        net, initial_marking, final_marking = heuristics_miner.apply(
            log,
            parameters={"dependency_thresh": config.heuristics_threshold},
        )
        label = f"Heuristics Miner (dependency={config.heuristics_threshold:.2f})"
        return net, initial_marking, final_marking, label

    raise ValueError(f"Unsupported miner: {config.miner_label}")


def render_petri_net(net, initial_marking, final_marking):
    if not PM4PY_AVAILABLE:
        return None, None

    gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    try:
        return gviz.pipe(format="png"), None
    except Exception:
        source = getattr(gviz, "source", None)
        return None, source


def get_model_activities(net, fallback_series: pd.Series) -> Tuple[str, ...]:
    try:
        if net is not None:
            activities = sorted({transition.label for transition in net.transitions if getattr(transition, "label", None)})
            if activities:
                return tuple(activities)
    except Exception:
        pass
    return tuple(sorted(fallback_series.dropna().astype(str).unique().tolist()))


def compute_conformance(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    metric_label: str,
    net,
    initial_marking,
    final_marking,
) -> pd.DataFrame:
    if not PM4PY_AVAILABLE:
        raise RuntimeError("pm4py is not installed.")

    log = to_pm4py_event_log(df, mapping.case_id, mapping.activity, mapping.timestamp)
    traces = list(log)

    if metric_label.startswith("Token-Based"):
        from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

        result = token_replay.apply(log, net, initial_marking, final_marking, variant=token_replay.Variants.TOKEN_REPLAY)
        rows = []
        for index, item in enumerate(result):
            rows.append(
                {
                    "case_id": traces[index].attributes.get("concept:name", f"case_{index}"),
                    "fitness": item.get("trace_fitness"),
                    "consumed_tokens": item.get("consumed_tokens"),
                    "produced_tokens": item.get("produced_tokens"),
                    "missing_tokens": item.get("missing_tokens"),
                    "remaining_tokens": item.get("remaining_tokens"),
                    "method": "Token-Based Replay",
                }
            )
        return pd.DataFrame(rows)

    alignments = None
    align_factory = None
    try:
        from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
    except Exception:
        try:
            from pm4py.algo.conformance.alignments import algorithm as alignments
        except Exception:
            from pm4py.algo.conformance.alignments import factory as align_factory

    variant = None
    if alignments is not None and hasattr(alignments, "Variants"):
        for name in [
            "VERSION_STATE_EQUATION_A_STAR",
            "A_STAR",
            "VERSION_TWEAKED",
            "DIJKSTRA",
        ]:
            if hasattr(alignments.Variants, name):
                variant = getattr(alignments.Variants, name)
                break

    if alignments is not None:
        alignment_result = (
            alignments.apply_log(log, net, initial_marking, final_marking, variant=variant)
            if variant is not None
            else alignments.apply_log(log, net, initial_marking, final_marking)
        )
    else:
        alignment_result = align_factory.apply(log, net, initial_marking, final_marking)

    rows = []
    for index, item in enumerate(alignment_result):
        fitness = item.get("fitness") if isinstance(item, dict) else None
        cost = item.get("cost") if isinstance(item, dict) else None
        rows.append(
            {
                "case_id": traces[index].attributes.get("concept:name", f"case_{index}"),
                "fitness": fitness,
                "cost": cost,
                "method": "Alignment-Based Replay",
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.title(APP_TITLE)
st.write(APP_DESCRIPTION)

with st.sidebar:
    st.header("⚙️ Settings")

    model_source = st.radio(
        "Model source",
        options=["Discover from current event log", "Load model specification (.txt)"],
        help=(
            "Discover a model from the current event log, or load a model specification TXT file "
            "containing transitions, start activities, and end activities."
        ),
    )

    miner_label = "Manual (select observed flows)"
    heuristics_threshold = 0.5

    if model_source == "Discover from current event log":
        miner_label = st.selectbox(
            "Discovery algorithm",
            options=[
                "Inductive Miner (recommended)",
                "Heuristics Miner",
                "Manual (select observed flows)",
            ],
            help=(
                "Inductive Miner creates sound, structured models. Heuristics Miner is useful on noisy logs. "
                "Manual mode lets you convert selected observed flows into a Petri net."
            ),
        )

        if miner_label.startswith("Heuristics"):
            heuristics_threshold = st.slider(
                "Heuristics dependency threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Higher values keep only stronger dependencies.",
            )

    show_full_dfg = st.checkbox("Show full-process DFG", value=False)
    show_section_dfgs = st.checkbox("Show section-specific DFGs (if available)", value=False)
    show_petri_net = st.checkbox("Attempt Petri net discovery", value=True)

    conformance_metric = st.selectbox(
        "Conformance metric",
        options=["Alignment-Based Replay (precise)", "Token-Based Replay (fast)"],
        help=(
            "Token-Based Replay is usually faster. Alignment-Based Replay is more precise but can be expensive on larger logs."
        ),
    )

    st.markdown("---")
    st.caption("Tip: If timestamp parsing fails, specify a format such as %Y-%m-%d %H:%M:%S.")


# -----------------------------------------------------------------------------
# File upload and column mapping
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload CSV event log", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

try:
    raw_df = load_csv(uploaded_file.getvalue())
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

if raw_df.empty:
    st.error("The uploaded CSV is empty.")
    st.stop()

with st.expander("🔍 Preview uploaded data", expanded=True):
    st.dataframe(raw_df.head(20), use_container_width=True)

st.subheader("🧩 Map event log columns")
columns = list(raw_df.columns)
case_default = _default_index(columns, CASE_SYNONYMS, 0)
activity_default = _default_index(columns, ACTIVITY_SYNONYMS, 1)
timestamp_default = _default_index(columns, TIMESTAMP_SYNONYMS, 2)

map_col1, map_col2, map_col3 = st.columns(3)
with map_col1:
    case_col = st.selectbox("Case ID column", options=columns, index=case_default)
with map_col2:
    activity_col = st.selectbox("Activity column", options=columns, index=activity_default)
with map_col3:
    timestamp_col = st.selectbox("Timestamp column", options=columns, index=timestamp_default)

timestamp_mode = st.radio("Timestamp parsing", options=["Auto-detect", "Provide a format"], horizontal=True)
timestamp_format = None
if timestamp_mode == "Provide a format":
    timestamp_format = st.text_input("Datetime format", value="%Y-%m-%d %H:%M:%S")

config = AppConfig(
    model_source=model_source,
    miner_label=miner_label,
    heuristics_threshold=heuristics_threshold,
    show_full_dfg=show_full_dfg,
    show_section_dfgs=show_section_dfgs,
    show_petri_net=show_petri_net,
    conformance_metric=conformance_metric,
    timestamp_mode=timestamp_mode,
    timestamp_format=timestamp_format,
)
mapping = ColumnMapping(case_id=case_col, activity=activity_col, timestamp=timestamp_col)

try:
    event_log_df, diagnostics = prepare_event_log(
        raw_df,
        mapping.case_id,
        mapping.activity,
        mapping.timestamp,
        config.timestamp_mode,
        config.timestamp_format,
    )
except Exception as exc:
    st.error(f"Failed to prepare the event log: {exc}")
    st.stop()

if event_log_df.empty:
    st.error("No valid events remain after parsing and cleaning. Please adjust the column mapping or timestamp format.")
    st.stop()

removed_total = diagnostics["original_rows"] - diagnostics["remaining_rows"]
if removed_total > 0:
    st.warning(
        f"Removed {removed_total:,} row(s): "
        f"{diagnostics['removed_empty_required']:,} with missing Case ID/Activity and "
        f"{diagnostics['removed_bad_timestamps']:,} with invalid timestamps."
    )

summary = compute_log_summary(event_log_df, mapping.case_id, mapping.timestamp)
section_summary = compute_section_summary(event_log_df, mapping.case_id)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
with metric_col1:
    st.metric("Events", f"{summary['events']:,}")
with metric_col2:
    st.metric("Cases", f"{summary['cases']:,}")
with metric_col3:
    st.metric("Avg events per case", f"{summary['avg_events_per_case']:.2f}")
with metric_col4:
    if summary["first_event"] is not None and summary["last_event"] is not None:
        span_days = (summary["last_event"] - summary["first_event"]).days
        st.metric("Observed span (days)", f"{span_days:,}")
    else:
        st.metric("Observed span (days)", "n/a")

_df_download("cleaned_event_log.csv", event_log_df, "⬇️ Download cleaned event log")


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
overview_tab, model_tab, conformance_tab, analytics_tab = st.tabs(
    ["Overview", "Process Model", "Conformance", "Analytics"]
)


# -----------------------------------------------------------------------------
# Overview tab
# -----------------------------------------------------------------------------
with overview_tab:
    st.subheader("📊 Log summary")
    span_col1, span_col2 = st.columns(2)
    with span_col1:
        st.write(f"**First event:** {summary['first_event']}")
    with span_col2:
        st.write(f"**Last event:** {summary['last_event']}")

    if section_summary is not None:
        st.subheader("📚 Section usage summary")
        sec_col1, sec_col2, sec_col3 = st.columns(3)
        with sec_col1:
            st.metric("Cases using only Section 1", f"{len(section_summary['only_section_1']):,}")
        with sec_col2:
            st.metric("Cases using only Section 2", f"{len(section_summary['only_section_2']):,}")
        with sec_col3:
            st.metric("Cases using both sections", f"{len(section_summary['both_sections']):,}")

        with st.expander("Show section case lists"):
            st.write("**Only Section 1:**", section_summary["only_section_1"])
            st.write("**Only Section 2:**", section_summary["only_section_2"])
            st.write("**Both sections:**", section_summary["both_sections"])

    dfg_df, _, _, _ = build_dfg_summary(event_log_df, mapping.case_id, mapping.activity, mapping.timestamp)

    st.subheader("⛓️ Directly-follows graph (DFG)")
    st.caption("Edge labels show frequency and average time between consecutive activities.")

    if config.show_full_dfg:
        st.markdown("#### Full process")
        full_dot = performance_dfg_dot(event_log_df, mapping.case_id, mapping.activity, mapping.timestamp)
        st.graphviz_chart(full_dot, use_container_width=True)

    if config.show_section_dfgs and "Section" in event_log_df.columns:
        st.markdown("#### Per section")
        for section_value in sorted(event_log_df["Section"].dropna().unique()):
            st.write(f"**Section {section_value}**")
            section_df = event_log_df.loc[event_log_df["Section"] == section_value].copy()
            section_dot = performance_dfg_dot(section_df, mapping.case_id, mapping.activity, mapping.timestamp)
            st.graphviz_chart(section_dot, use_container_width=True)

    with st.expander("Show DFG transition table"):
        dfg_table = dfg_df.copy()
        dfg_table["avg_duration"] = dfg_table["avg_duration_seconds"].map(_human_duration)
        dfg_table["median_duration"] = dfg_table["median_duration_seconds"].map(_human_duration)
        st.dataframe(
            dfg_table[["from_activity", "to_activity", "frequency", "avg_duration", "median_duration"]],
            use_container_width=True,
        )
        _df_download("dfg_transitions.csv", dfg_table, "⬇️ Download DFG transitions")


# -----------------------------------------------------------------------------
# Process model tab
# -----------------------------------------------------------------------------
net = None
initial_marking = None
final_marking = None
model_name = None

model_spec = None
model_dfg_dict: Dict[Tuple[str, str], int] = {}
model_start_acts: Dict[str, int] = {}
model_end_acts: Dict[str, int] = {}

with model_tab:
    st.subheader("🧭 Process model")

    uploaded_model_txt = None
    if config.model_source == "Load model specification (.txt)":
        uploaded_model_txt = st.file_uploader(
            "📥 Upload model specification TXT",
            type=["txt"],
            help="Upload a TXT file containing transitions, start activities, and end activities in JSON text format.",
        )

    if not PM4PY_AVAILABLE:
        st.info("Install pm4py to enable Petri net discovery and conformance checking: `pip install pm4py`.")
    elif not config.show_petri_net:
        st.info("Petri net discovery/loading is disabled in the sidebar.")
    else:
        try:
            if config.model_source == "Load model specification (.txt)":
                if uploaded_model_txt is None:
                    st.info("Upload a model TXT file to build the model.")
                else:
                    model_spec = load_model_spec_from_txt(uploaded_model_txt.getvalue())
                    model_name = str(model_spec.get("model_name", "Loaded model specification")).strip() or "Loaded model specification"

                    (
                        net,
                        initial_marking,
                        final_marking,
                        model_dfg_dict,
                        model_start_acts,
                        model_end_acts,
                    ) = build_petri_net_from_model_spec(model_spec)

                    st.success(f"Model specification loaded: {model_name}")

                    with st.expander("Show loaded model specification"):
                        st.json(model_spec)

            else:
                if config.miner_label == "Manual (select observed flows)":
                    dfg_df, activity_counts, start_activities, end_activities = build_dfg_summary(
                        event_log_df,
                        mapping.case_id,
                        mapping.activity,
                        mapping.timestamp,
                    )
                    if dfg_df.empty:
                        st.warning("No directly-follows relations were found, so a Petri net cannot be built.")
                    else:
                        st.caption("Select the observed flows and start/end activities that should form the model.")
                        max_frequency = int(dfg_df["frequency"].max())
                        min_frequency = st.slider(
                            "Minimum transition frequency",
                            min_value=1,
                            max_value=max_frequency,
                            value=min(3, max_frequency),
                            help="Only transitions with at least this frequency are shown.",
                        )
                        candidates = dfg_df.loc[dfg_df["frequency"] >= min_frequency].copy()
                        candidates["label"] = candidates.apply(
                            lambda row: f"{row['from_activity']} → {row['to_activity']} (freq={int(row['frequency'])})",
                            axis=1,
                        )

                        selected_labels = st.multiselect(
                            "Transitions to include",
                            options=candidates["label"].tolist(),
                            default=candidates["label"].head(min(15, len(candidates))).tolist(),
                        )

                        selected = candidates.loc[candidates["label"].isin(selected_labels)].copy()
                        selected_activities = sorted(
                            set(selected["from_activity"].tolist()) | set(selected["to_activity"].tolist())
                        )

                        default_starts = [activity for activity in start_activities if activity in selected_activities] or list(start_activities)
                        default_ends = [activity for activity in end_activities if activity in selected_activities] or list(end_activities)

                        selected_starts = st.multiselect(
                            "Start activities",
                            options=sorted(activity_counts.keys()),
                            default=sorted(default_starts),
                        )
                        selected_ends = st.multiselect(
                            "End activities",
                            options=sorted(activity_counts.keys()),
                            default=sorted(default_ends),
                        )

                        if selected.empty:
                            st.info("Select at least one transition to build a Petri net.")
                        else:
                            model_dfg_dict = {
                                (row.from_activity, row.to_activity): int(row.frequency)
                                for row in selected.itertuples(index=False)
                            }
                            model_start_acts = {
                                activity: int(start_activities.get(activity, 1))
                                for activity in selected_starts
                            }
                            model_end_acts = {
                                activity: int(end_activities.get(activity, 1))
                                for activity in selected_ends
                            }

                            net, initial_marking, final_marking = convert_dfg_to_petri(
                                model_dfg_dict,
                                model_start_acts,
                                model_end_acts,
                            )
                            model_name = "Manual model from selected flows"

                            model_spec = build_model_spec(
                                model_name=model_name,
                                dfg_dict=model_dfg_dict,
                                start_acts=model_start_acts,
                                end_acts=model_end_acts,
                            )

                            st.markdown("#### Selected flows")
                            st.dataframe(
                                selected[["from_activity", "to_activity", "frequency"]],
                                use_container_width=True,
                            )

                else:
                    # Discover Petri net in the original way
                    net, initial_marking, final_marking, model_name = discover_petri_model(event_log_df, mapping, config)

                    # Build a saveable model specification from the current event log DFG
                    dfg_df, _, start_activities, end_activities = build_dfg_summary(
                        event_log_df,
                        mapping.case_id,
                        mapping.activity,
                        mapping.timestamp,
                    )

                    model_dfg_dict = {
                        (row.from_activity, row.to_activity): int(row.frequency)
                        for row in dfg_df.itertuples(index=False)
                    }
                    model_start_acts = {str(activity): int(count) for activity, count in start_activities.items()}
                    model_end_acts = {str(activity): int(count) for activity, count in end_activities.items()}

                    model_spec = build_model_spec(
                        model_name=model_name,
                        dfg_dict=model_dfg_dict,
                        start_acts=model_start_acts,
                        end_acts=model_end_acts,
                    )

            if net is not None and initial_marking is not None and final_marking is not None:
                st.success(f"Model ready: {model_name}")

                png_bytes, graphviz_source = render_petri_net(net, initial_marking, final_marking)
                if png_bytes is not None:
                    st.image(png_bytes, caption=model_name, use_container_width=True)
                elif graphviz_source is not None:
                    st.graphviz_chart(graphviz_source, use_container_width=True)
                else:
                    st.info("The model was created, but no renderer was available for preview.")

                if model_spec is not None:
                    st.download_button(
                        label="⬇️ Download model specification (.txt)",
                        data=export_model_spec_txt_bytes(model_spec),
                        file_name="process_model_spec.txt",
                        mime="text/plain",
                    )

                    with st.expander("Show model specification that will be saved"):
                        st.json(model_spec)

            else:
                st.info("No Petri net is available yet.")

        except Exception as exc:
            st.error(f"Model discovery/loading failed: {exc}")
            st.info(
                "Troubleshooting tips: verify the TXT structure, verify timestamp parsing, "
                "or reduce the number of activities in manual mode."
            )


# -----------------------------------------------------------------------------
# Conformance tab
# -----------------------------------------------------------------------------
with conformance_tab:
    st.subheader("✔️ Conformance checking")
    st.write(
        "Conformance checking compares the discovered model to the observed event log. "
        "A fitness score closer to 1.0 means the trace fits the model well."
    )

    if not PM4PY_AVAILABLE:
        st.info("Install pm4py to enable conformance checking.")
    elif net is None or initial_marking is None or final_marking is None:
        st.info("Discover a Petri net in the Process Model tab first.")
    else:
        case_count = event_log_df[mapping.case_id].nunique()
        if config.conformance_metric.startswith("Alignment") and case_count > 1000:
            st.warning(
                "Alignment-based replay can be slow on large logs. Consider Token-Based Replay for a quicker review."
            )

        try:
            fitness_df = compute_conformance(
                event_log_df,
                mapping,
                config.conformance_metric,
                net,
                initial_marking,
                final_marking,
            )

            if fitness_df.empty or not fitness_df["fitness"].notna().any():
                st.info("No fitness values were produced. Try the other conformance metric.")
            else:
                avg_fitness = fitness_df["fitness"].mean()
                median_fitness = fitness_df["fitness"].median()
                worst_fitness = fitness_df["fitness"].min()

                fit_col1, fit_col2, fit_col3 = st.columns(3)
                with fit_col1:
                    st.metric("Average fitness", f"{avg_fitness:.3f}")
                with fit_col2:
                    st.metric("Median fitness", f"{median_fitness:.3f}")
                with fit_col3:
                    st.metric("Worst case fitness", f"{worst_fitness:.3f}")

                threshold = st.slider("Fitness pass threshold", 0.0, 1.0, 0.90, 0.01)
                passed = int((fitness_df["fitness"] >= threshold).sum())
                st.metric("Cases meeting threshold", f"{passed}/{len(fitness_df)} ({_percentage(passed, len(fitness_df)):.0f}%)")

                st.markdown("#### Fitness distribution")
                hist = (
                    alt.Chart(fitness_df.dropna(subset=["fitness"]))
                    .mark_bar(color="#4F46E5", opacity=0.75)
                    .encode(
                        x=alt.X("fitness:Q", bin=alt.Bin(maxbins=30), title="Fitness"),
                        y=alt.Y("count():Q", title="Cases"),
                        tooltip=[alt.Tooltip("count():Q", title="Cases")],
                    )
                )
                st.altair_chart(hist, use_container_width=True)

                sorted_fitness = fitness_df.sort_values("fitness", ascending=True).reset_index(drop=True)
                line = (
                    alt.Chart(sorted_fitness)
                    .mark_line(point=True, color="#2563EB")
                    .encode(
                        x=alt.X("case_id:N", sort=None, title="Case ID"),
                        y=alt.Y("fitness:Q", title="Fitness"),
                        tooltip=["case_id:N", alt.Tooltip("fitness:Q", format=".3f")],
                    )
                )
                st.altair_chart(line, use_container_width=True)

                with st.expander("Per-case conformance results"):
                    st.dataframe(fitness_df, use_container_width=True)
                    _df_download("conformance_results.csv", fitness_df, "⬇️ Download conformance results")
        except Exception as exc:
            st.error(f"Conformance checking failed: {exc}")
            st.info("If the problem persists, try another miner or update pm4py.")


# -----------------------------------------------------------------------------
# Analytics tab
# -----------------------------------------------------------------------------
with analytics_tab:
    st.header("🔬 Analytics")

    case_summary = compute_case_summary(event_log_df, mapping.case_id, mapping.activity, mapping.timestamp)
    variants_df = compute_variants(event_log_df, mapping.case_id, mapping.activity)
    activity_freq_df, activity_service_df = compute_activity_summary(
        event_log_df,
        mapping.case_id,
        mapping.activity,
        mapping.timestamp,
    )
    transition_pairs_df = compute_transition_pairs(
        event_log_df,
        mapping.case_id,
        mapping.activity,
        mapping.timestamp,
    )
    transition_stats_df = compute_transition_stats(transition_pairs_df)
    wip_df = compute_wip_series(event_log_df, mapping.case_id, mapping.timestamp)

    # Case-level analytics
    st.subheader("📦 Case-level analytics")
    case_col1, case_col2, case_col3 = st.columns(3)
    with case_col1:
        st.metric("Avg events per case", f"{case_summary['events_per_case'].mean():.2f}")
    with case_col2:
        st.metric(
            "Median events per case",
            f"{case_summary['events_per_case'].median():.0f}",
        )
    with case_col3:
        avg_throughput = case_summary["throughput_seconds"].mean()
        st.metric("Avg throughput", _human_duration(avg_throughput))

    event_count_distribution = (
        case_summary["events_per_case"].value_counts().sort_index().rename_axis("events_per_case").reset_index(name="case_count")
    )
    events_chart = (
        alt.Chart(event_count_distribution)
        .mark_bar(color="#4F46E5")
        .encode(
            x=alt.X("events_per_case:O", title="Events per case", sort=None),
            y=alt.Y("case_count:Q", title="Cases"),
            tooltip=["events_per_case:Q", "case_count:Q"],
        )
    )
    st.altair_chart(events_chart, use_container_width=True)

    with st.expander("Show case-level table"):
        display_case_summary = case_summary.copy()
        display_case_summary["throughput"] = display_case_summary["throughput_seconds"].map(_human_duration)
        st.dataframe(
            display_case_summary[["case_id", "events_per_case", "start", "end", "throughput"]],
            use_container_width=True,
        )
        _df_download("case_summary.csv", display_case_summary, "⬇️ Download case summary")

    # Variant analysis
    st.subheader("🧬 Variant analysis")
    top_variants = variants_df.loc[variants_df["cumulative_pct"] <= 80.0].copy()
    if top_variants.empty and not variants_df.empty:
        top_variants = variants_df.head(1).copy()

    st.write(f"Showing {len(top_variants):,} variant(s) needed to cover roughly 80% of the dataset.")
    st.dataframe(
        top_variants[["variant_display", "number_of_cases", "cumulative_cases", "cumulative_pct"]],
        use_container_width=True,
    )

    top_variant_plot = top_variants.reset_index(drop=True).copy()
    top_variant_plot["variant_no"] = top_variant_plot.index.astype(str)

    variant_bar = (
        alt.Chart(top_variant_plot)
        .mark_bar(color="#4F46E5")
        .encode(
            x=alt.X("variant_no:N", title="Variant #", sort=None),
            y=alt.Y("number_of_cases:Q", title="Cases"),
            tooltip=[
                alt.Tooltip("variant_no:N", title="Variant #"),
                alt.Tooltip("number_of_cases:Q", title="Cases"),
                alt.Tooltip("variant_display:N", title="Variant"),
            ],
        )
    )
    variant_line = (
        alt.Chart(top_variant_plot)
        .mark_line(color="#DC2626", point=True)
        .encode(
            x=alt.X("variant_no:N", sort=None),
            y=alt.Y("cumulative_pct:Q", title="Cumulative %"),
            tooltip=[alt.Tooltip("cumulative_pct:Q", format=".1f")],
        )
    )
    st.altair_chart(variant_bar + variant_line, use_container_width=True)
    _df_download("variants.csv", variants_df, "⬇️ Download variant analysis")

    # Activity analytics
    st.subheader("🏷️ Activity analytics")
    act_col1, act_col2 = st.columns(2)
    with act_col1:
        st.write("**Top activities by frequency**")
        st.dataframe(activity_freq_df.head(DEFAULT_TOP_N), use_container_width=True)
    with act_col2:
        st.write("**Activities with longest average time to next step**")
        st.dataframe(
            activity_service_df.sort_values("Average time to next activity (days)", ascending=False).head(DEFAULT_TOP_N),
            use_container_width=True,
        )
    _df_download("activity_frequency.csv", activity_freq_df, "⬇️ Download activity frequency")
    _df_download("activity_service_time.csv", activity_service_df, "⬇️ Download activity service-time summary")

    # Skip analysis
    st.subheader("🚫 Skipped activities")
    if section_summary is not None:
        only_common_cases = st.checkbox(
            "Only include cases that appear in more than one section",
            value=False,
            help="Useful when Section 1 and Section 2 represent different systems for the same business object.",
        )
    else:
        only_common_cases = False

    if only_common_cases and section_summary is not None:
        analysis_df = event_log_df.loc[event_log_df[mapping.case_id].isin(section_summary["common_cases"])].copy()
        st.caption(f"Skip analysis is restricted to {analysis_df[mapping.case_id].nunique():,} shared case(s).")
    else:
        analysis_df = event_log_df

    activity_universe = get_model_activities(net, analysis_df[mapping.activity])
    skip_df = compute_skip_summary(analysis_df, mapping.case_id, mapping.activity, activity_universe)
    top_k = st.slider(
        "Number of skipped activities to plot",
        min_value=5,
        max_value=max(5, len(skip_df) if not skip_df.empty else 5),
        value=min(15, max(5, len(skip_df) if not skip_df.empty else 5)),
    )

    skip_chart = (
        alt.Chart(skip_df.head(top_k))
        .mark_bar(color="#EF4444")
        .encode(
            y=alt.Y("Activity:N", sort="-x", title="Activity"),
            x=alt.X("Cases skipping:Q", title="Cases skipping"),
            tooltip=["Activity:N", alt.Tooltip("Cases skipping:Q"), alt.Tooltip("Share skipping (%):Q", format=".1f")],
        )
    )
    st.altair_chart(skip_chart, use_container_width=True)

    skip_table = skip_df.rename(
        columns={
            "Cases skipping": "Total skips",
            "Share skipping (%)": "Skip share (%)",
            "Cases present": "Cases executing activity",
            "Total cases": "Total cases",
        }
    )
    st.dataframe(
        skip_table[["Activity", "Total skips", "Skip share (%)", "Cases executing activity", "Total cases"]],
        use_container_width=True,
    )

    with st.expander("Explore case IDs skipping each activity"):
        for row in skip_df.itertuples(index=False):
            activity = row[0]
            case_ids = row[5]
            with st.expander(f"{activity} — {len(case_ids)} case(s) skipping"):
                if not case_ids:
                    st.caption("All cases include this activity.")
                else:
                    st.write(", ".join(map(str, case_ids)))

    long_skip_df = skip_df[["Activity", "Missing case IDs"]].explode("Missing case IDs").rename(
        columns={"Missing case IDs": "Case ID"}
    )
    _df_download("cases_skipping_activities.csv", long_skip_df, "⬇️ Download skipped-activity pairs")

    # Transition analytics
    st.subheader("➡️ Transition analytics")
    st.write("**Slowest transitions by average duration**")
    transition_display = transition_stats_df.copy()
    st.dataframe(
        transition_display[
            [
                "from_activity",
                "to_activity",
                "count",
                "avg_days",
                "p50_days",
                "p90_days",
                "p95_days",
            ]
        ].head(DEFAULT_TOP_N),
        use_container_width=True,
    )

    if not transition_pairs_df.empty:
        transition_options = (
            transition_pairs_df[["from_activity", "to_activity"]]
            .drop_duplicates()
            .assign(label=lambda frame: frame["from_activity"] + " → " + frame["to_activity"])
        )
        selected_transition = st.selectbox("Choose a transition", transition_options["label"].tolist())
        selected_from, selected_to = selected_transition.split(" → ", 1)
        selected_transition_df = transition_pairs_df.loc[
            (transition_pairs_df["from_activity"] == selected_from)
            & (transition_pairs_df["to_activity"] == selected_to)
        ].copy()
        selected_transition_df["days"] = selected_transition_df["delta_seconds"] / (24 * 3600)

        hist = (
            alt.Chart(selected_transition_df)
            .mark_bar(color="#4F46E5", opacity=0.75)
            .encode(
                x=alt.X("days:Q", bin=alt.Bin(maxbins=40), title="Duration (days)"),
                y=alt.Y("count():Q", title="Frequency"),
                tooltip=[alt.Tooltip("count():Q", title="Frequency")],
            )
        )
        st.altair_chart(hist, use_container_width=True)
        _df_download("transition_pairs.csv", transition_pairs_df, "⬇️ Download transition pairs")
        _df_download("transition_stats.csv", transition_stats_df, "⬇️ Download transition statistics")

    # Work in progress
    st.subheader("📈 Work-in-progress (WIP) over time")
    if wip_df.empty:
        st.info("Not enough information is available to compute WIP.")
    else:
        if len(wip_df) > 5000:
            step = max(1, len(wip_df) // 2000)
            wip_plot_df = wip_df.iloc[::step, :].copy()
        else:
            wip_plot_df = wip_df.copy()

        wip_chart = (
            alt.Chart(wip_plot_df)
            .mark_line(color="#2563EB")
            .encode(
                x=alt.X("timestamp:T", title="Time"),
                y=alt.Y("wip:Q", title="Active cases"),
                tooltip=["timestamp:T", "wip:Q"],
            )
        )
        st.altair_chart(wip_chart, use_container_width=True)
        _df_download("wip_timeseries.csv", wip_df, "⬇️ Download WIP time series")
