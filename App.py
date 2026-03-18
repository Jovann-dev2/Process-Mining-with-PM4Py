import math
from datetime import timedelta

import altair as alt
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import streamlit as st
import sys

import tempfile
import zipfile
import pickle
from brokenaxes import brokenaxes
from typing import Tuple, Optional

# Core process mining (Python)
# If pm4py import fails, we degrade gracefully (we'll still show DFG + stats)
try:
    import pm4py
    from pm4py import discover_petri_net_inductive
    from pm4py.objects.conversion.log import converter as log_converter
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
    from pm4py.visualization.petri_net import visualizer as pn_visualizer
    PM4PY_AVAILABLE = True
except Exception as e:
    PM4PY_AVAILABLE = False

st.set_page_config(page_title="Process Mining App", layout="wide")

st.title("🔎 A Visualization of Process Mining using PM4Py")
st.write(
    "Upload an event log below. The event log must be in CSV format, and it must have at least the following three columns: a 'Case-ID' column, an 'Activity' column, and a 'Timestamp' column. "
    "The names need not be as presented; you can select which columns conform to the presented structure. "
)

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    model_source = "Discover from event log"
    #model_source = st.radio(
    #    "Model source",
    #    ["Discover from event log", "Load stored PNML"],
    #    help="Either mine a model from the uploaded CSV, or load a previously saved PNML Petri net."
    #)

    if model_source == "Discover from event log":
        miner = st.selectbox(
            "Discovery algorithm",
            ["Manual (select flows)", "Inductive Miner (recommended algorithm)", "Heuristics Miner"],
            help=(
                "Inductive Miner produces sound, block-structured models and is robust to noise. \n"
                "Heuristics Miner is good for noisy logs and uses dependency thresholds. \n"
                "Manual lets you pick which observed flows should be in the model."
            ),
        )
        if miner.startswith("Heuristics"):
            hm_threshold = st.slider(
                "Heuristics dependency threshold", 0.0, 1.0, 0.5, 0.05,
                help="A higher threshold relates to a more complex and inclusive model."
            )
        else:
            hm_threshold = None
    else:
        miner = None
        hm_threshold = None

    show_perf = st.checkbox(
        "Show Graph of Observed Process", value=True,
        help="Computed directly from your data; no external binaries needed."
    )

    show_petri = st.checkbox(
        "Attempt Petri Net Visualization (requires pm4py)", value=True
    )

    # Choose conformance metric
    fitness_metric = st.selectbox(
        "Conformance Metric",
        ["Alignment-Based Replay (precise)", "Token-Based Replay (fast)"],
        help=(
            "Alignment-based replay uses A* search to find optimal alignments and yields precise conformance at higher computational cost. "
            "Token-based replay is faster and approximate. "
        )
    )

    st.markdown("---")
    st.caption("Tip: If timestamps are not parsed automatically, provide a format, e.g. `%Y-%m-%d %H:%M:%S`")

# ---------------------------
# File upload
# ---------------------------

@st.cache_data
def load_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))

uploaded = st.file_uploader("📤 Upload CSV event log", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to get started.")
    st.stop()

# Read CSV
try:
    df = load_csv(uploaded.getvalue())
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

if df.empty:
    st.error("The CSV is empty.")
    st.stop()

st.subheader("🔍 Preview")
st.write(
    "Below is a preview of the uploaded dataset (20 rows). "
)
st.dataframe(df.head(20), width='stretch')

# ---------------------------
# PNML Import/Export Helpers (version-safe, with fallback)
# ---------------------------

def _export_pnml_via_read_write(net, im, fm, path) -> bool:
    """
    Try pm4py >= 2.7 style: pm4py.read_write.pnml.exporter.apply(...)
    Returns True if successful.
    """
    try:
        from pm4py.read_write.pnml import exporter as pnml_exporter
        # Most versions: exporter.apply(net, im, fm, file_path)
        try:
            pnml_exporter.apply(net, im, fm, path)
        except TypeError:
            # Some expect named args
            pnml_exporter.apply(net=net, initial_marking=im, final_marking=fm, file_path=path)
        return True
    except Exception:
        return False

def _export_pnml_via_objects(net, im, fm, path) -> bool:
    """
    Try older style: pm4py.objects.petri.exporter[.variants].pnml
    Returns True if successful.
    """
    # Most common old path
    try:
        from pm4py.objects.petri.exporter import exporter as pnml_exporter
        # API differences: some accept (net, im, path, final_marking=fm)
        try:
            pnml_exporter.apply(net, im, path, final_marking=fm)
        except TypeError:
            # Other versions
            pnml_exporter.apply(net=net, initial_marking=im, file_path=path, final_marking=fm)
        return True
    except Exception:
        pass

    # Variant submodule
    try:
        from pm4py.objects.petri.exporter.variants import pnml as pnml_exporter
        pnml_exporter.apply(net, im, path, final_marking=fm)
        return True
    except Exception:
        return False

def export_petri_to_pnml_bytes(net, im, fm) -> bytes:
    """
    Export Petri net to PNML (bytes). Tries multiple pm4py APIs.
    Raises if PNML export fails (so caller can fallback to Pickle).
    """
    if net is None or im is None or fm is None:
        raise ValueError("Cannot export PNML: net/im/fm are not all available.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pnml") as tmp:
        path = tmp.name
    try:
        ok = _export_pnml_via_read_write(net, im, fm, path) or _export_pnml_via_objects(net, im, fm, path)
        if not ok:
            raise ImportError(
                "PNML exporter not available in this pm4py build "
                "(neither pm4py.read_write.pnml nor pm4py.objects.petri.exporter found)."
            )
        with open(path, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def _import_pnml_via_read_write(path):
    try:
        from pm4py.read_write.pnml import importer as pnml_importer
        result = pnml_importer.apply(path)
        # Many versions return (net, im, fm)
        if isinstance(result, tuple) and len(result) >= 3:
            return result[0], result[1], result[2]
        if isinstance(result, dict):
            return result.get("net"), result.get("initial_marking"), result.get("final_marking")
        # Some versions return a namedtuple-like
        try:
            return result.net, result.initial_marking, result.final_marking
        except Exception:
            pass
    except Exception:
        pass
    return None


def _import_pnml_via_objects(path):
    try:
        from pm4py.objects.petri.importer import importer as pnml_importer
        res = pnml_importer.apply(path)
        if isinstance(res, tuple) and len(res) >= 3:
            return res[0], res[1], res[2]
        if isinstance(res, dict):
            return res.get("net"), res.get("initial_marking"), res.get("final_marking")
        try:
            return res.net, res.initial_marking, res.final_marking
        except Exception:
            pass
    except Exception:
        pass

    try:
        from pm4py.objects.petri.importer.variants import pnml as pnml_importer
        res = pnml_importer.apply(path)
        if isinstance(res, tuple) and len(res) >= 3:
            return res[0], res[1], res[2]
        if isinstance(res, dict):
            return res.get("net"), res.get("initial_marking"), res.get("final_marking")
        try:
            return res.net, res.initial_marking, res.final_marking
        except Exception:
            pass
    except Exception:
        pass
    return None


def import_petri_from_pnml_bytes(pnml_bytes: bytes):
    """
    Import Petri net from PNML bytes.
    Returns (net, im, fm) or raises on failure.
    """
    if not pnml_bytes:
        raise ValueError("Empty PNML bytes.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pnml") as tmp:
        tmp.write(pnml_bytes)
        path = tmp.name
    try:
        triple = _import_pnml_via_read_write(path) or _import_pnml_via_objects(path)
        if not triple:
            raise ImportError(
                "PNML importer not available in this pm4py build "
                "(neither pm4py.read_write.pnml nor pm4py.objects.petri.importer found)."
            )
        net, im, fm = triple
        if net is None or im is None or fm is None:
            raise RuntimeError("PNML did not contain net/initial/final markings.")
        return net, im, fm
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


# ---------------------------
# Optional Fallback: ZIP(Pickle) for environments without PNML support
# ---------------------------
def export_petri_to_pickle_zip_bytes(net, im, fm) -> bytes:
    """
    Fallback serialization: zip with three pickles (net.pkl, im.pkl, fm.pkl).
    Not portable outside Python/pm4py, but works when PNML isn't available.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("net.pkl", pickle.dumps(net))
        zf.writestr("initial_marking.pkl", pickle.dumps(im))
        zf.writestr("final_marking.pkl", pickle.dumps(fm))
    return buf.getvalue()

def import_petri_from_pickle_zip_bytes(zip_bytes: bytes):
    buf = io.BytesIO(zip_bytes)
    with zipfile.ZipFile(buf, mode="r") as zf:
        net = pickle.loads(zf.read("net.pkl"))
        im = pickle.loads(zf.read("initial_marking.pkl"))
        fm = pickle.loads(zf.read("final_marking.pkl"))
    return net, im, fm

# ---------------------------
# Column mapping
# ---------------------------
st.subheader("🧩 Map Event Log Columns")
st.write(
    "Choose which columns represent 'Case ID', 'Activity', and 'Timestamp'. "
    "Also, you can provide a format for the 'Timestamp' column if you know the certain format. "
)
# Build a quick lookup for lowercase names
cols = list(df.columns)
cols_lower = [c.lower() for c in cols]

def find_index_or_default(target: str, default_idx: int, num_cols: int) -> int:
    """
    Return the index of the column whose lowercase name matches `target`,
    otherwise return a clamped default index.
    """
    if target in cols_lower:
        return cols_lower.index(target)
    # Clamp default to the valid range [0, num_cols-1]
    return min(default_idx, max(0, num_cols - 1))

# Defaults:
# - Case ID: keep as first column (index 0) unless a case_id-like name exists
# - Activity: default to 2nd column (index 1) unless "activity" exists
# - Timestamp: default to 3rd column (index 2) unless "timestamp" exists
case_idx = find_index_or_default("case_id", 0, len(cols))
act_idx  = find_index_or_default("activity", 1, len(cols))
time_idx = find_index_or_default("timestamp", 2, len(cols))

col_case = st.selectbox("Case ID column", options=cols, index=case_idx)
col_act  = st.selectbox("Activity column", options=cols, index=act_idx)
col_time = st.selectbox("Timestamp column", options=cols, index=time_idx)

time_fmt_mode = st.radio("Timestamp parsing", ["Auto-detect", "Provide a format"])
time_fmt = None
if time_fmt_mode == "Provide a format":
    time_fmt = st.text_input("Datetime format (e.g. %Y-%m-%d %H:%M:%S)", value="%Y-%m-%d %H:%M:%S")

@st.cache_data
def preprocess_data(df, col_case, col_act, col_time, time_fmt_mode, time_fmt):
    work = df.copy()

    if time_fmt_mode == "Auto-detect":
        work[col_time] = pd.to_datetime(work[col_time], errors="coerce")
    else:
        work[col_time] = pd.to_datetime(work[col_time], format=time_fmt, errors="coerce")

    work = work[work[col_time].notna()].copy()
    work.sort_values([col_case, col_time], inplace=True)
    return work

# Prepare working dataframe
work = preprocess_data(df, col_case, col_act, col_time, time_fmt_mode, time_fmt)
# Parse timestamps
try:
    if time_fmt_mode == "Auto-detect":
        work[col_time] = pd.to_datetime(work[col_time], errors="coerce", utc=False)
    else:
        work[col_time] = pd.to_datetime(work[col_time], format=time_fmt, errors="coerce", utc=False)
except Exception as e:
    st.error(f"Timestamp parsing failed: {e}")
    st.stop()

# Drop events with unparseable timestamps
bad_ts = work[col_time].isna().sum()
if bad_ts > 0:
    st.warning(f"Dropping {bad_ts} rows with unparseable timestamps.")
    work = work[work[col_time].notna()].copy()

# Sort by case and time
work.sort_values([col_case, col_time], inplace=True)

# ---------------------------
# KPI & basic stats
# ---------------------------
st.subheader("📊 Log Summary")
st.write(
    "Below is the number of events that occurred, together with the number of cases (Case IDs) present, in the data. "
)
n_events = len(work)
n_cases = work[col_case].nunique()
st.metric("Events", f"{n_events:,}")
st.metric("Cases", f"{n_cases:,}")

# ---------------------------
# Section-based case metrics
# ---------------------------
if "Section" in work.columns:

    st.subheader("📚 Section Usage Summary")

    # For each case, determine the set of Sections used
    col_case2 = "ccc_id"
    case_sections = (
        work.groupby([col_case], dropna=False)["Section"]
            .apply(lambda x: set(x.unique()))
            .reset_index(name="sections_used")
    )

    col_count1 = len(work[col_case].dropna().unique())
    col_count2 = len(work[col_case2].dropna().unique())

    # Count rows where both columns are non-empty / non-blank
    filtered = work.loc[work[col_case].notna() & work[col_case2].notna()]
    overlap_count = filtered.drop_duplicates(subset=[col_case]).shape[0]

    colA, colB = st.columns(2)
    with colA:
        st.metric(
            "Number of contracts with unique Coupa Core IDs",
            f"{col_count1}"
        )
    with colB:
        st.metric(
            "Number of contracts with corresponding IDs over CLM and Coupa Core",
            f"{overlap_count}"
        )

    # Categorize cases
    only_1 = (case_sections[case_sections["sections_used"] == {1}]).dropna()
    only_2 = (case_sections[case_sections["sections_used"] == {2}]).dropna()
    both   = case_sections[case_sections["sections_used"].apply(lambda s: s == {1, 2})]

    n_only_1 = len(only_1)
    n_only_2 = len(only_2)
    n_both   = len(both)

    #pct_only_1 = n_only_1 / n_cases * 100 if n_cases > 0 else 0
    #pct_only_2 = n_only_2 / n_cases * 100 if n_cases > 0 else 0
    #pct_both   = n_both   / n_cases * 100 if n_cases > 0 else 0

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric(
            "Contracts using only CLM",
            f"{n_only_1}"
        )
    with colB:
        st.metric(
            "Contracts using only Coupa Core",
            f"{n_only_2}"
        )
    with colC:
        st.metric(
            "Contracts using both CLM and Coupa Core",
            f"{n_both}"
        )

    with st.expander("Details"):
        st.write("Contracts using only CLM:", only_1[col_case].tolist())
        st.write("Contracts using only Coupa Core:", only_2[col_case].tolist())
        st.write("Contracts using both Sections:", both[col_case].tolist())

    st.write(filtered)

# ---------------------------
# Performance DFG (client-side DOT render; no graphviz binary needed)
@st.cache_data
def cached_performance_dfg(work, case_col, act_col, time_col):
    return performance_dfg_dot(work, case_col, act_col, time_col)

# ---------------------------
def performance_dfg_dot(df_, case_col, act_col, time_col, max_nodes=50, max_edges=200):
    """
    Build a DOT graph showing activities as nodes and transitions as edges.
    Edge labels show: frequency | avg duration.
    Render-friendly limits for very large logs.
    """
    # Node counts
    node_counts = df_[act_col].value_counts().to_dict()

    # Build edges: for each case, consecutive pairs
    edges = {}  # (a,b) -> [count, total_duration_seconds]
    for _, g in df_.groupby(case_col, sort=False):
        g = g.sort_values(time_col)
        acts = g[act_col].tolist()
        times = g[time_col].tolist()
        for i in range(len(acts) - 1):
            a, b = acts[i], acts[i+1]
            dur = (times[i+1] - times[i]).total_seconds() if (times[i+1] and times[i]) else np.nan
            key = (a, b)
            if key not in edges:
                edges[key] = [0, 0.0, 0]  # count, total_dur, dur_count
            edges[key][0] += 1
            if not np.isnan(dur):
                edges[key][1] += float(dur)
                edges[key][2] += 1

    # Limit size (keep most frequent nodes/edges)
    top_nodes = dict(sorted(node_counts.items(), key=lambda x: x[1], reverse=True)[:max_nodes])
    # Filter edges to only edges where both nodes are in top_nodes
    edges = {k: v for k, v in edges.items() if k[0] in top_nodes and k[1] in top_nodes}
    # Keep top edges by frequency
    edges = dict(sorted(edges.items(), key=lambda x: x[1][0], reverse=True)[:max_edges])

    # DOT Graph
    def safe(s: str) -> str:
        return s.replace('"', '\\"')

    lines = ['digraph G {', '  rankdir=LR;', '  node [shape=box, style="rounded,filled", color="#4B5563", fillcolor="#E5E7EB"];']

    # Nodes with counts
    for node, cnt in top_nodes.items():
        lines.append(f'  "{safe(node)}" [label="{safe(node)}\\n({cnt} events)"];')

    # Edges with freq and avg duration
    for (a, b), (cnt, total_dur, dur_count) in edges.items():
        avg_dur = (total_dur / dur_count) if dur_count > 0 else None
        dur_label = str(timedelta(seconds=int(avg_dur))) if avg_dur is not None else "n/a"
        lines.append(
            f'  "{safe(a)}" -> "{safe(b)}" [label="{cnt} | {dur_label}", color="#2563EB"];'
        )

    lines.append("}")
    return "\n".join(lines)

if show_perf:
    st.subheader("⛓️ Performance Directly-Follows Graphs (DFGs)")

    if "Section" in work.columns:
        st.write(
            "The dataset contains multiple Sections. "
            "A separate performance DFG is shown for each Section."
        )

        # Get unique section labels (e.g., '1', '2')
        for sec in sorted(work["Section"].dropna().unique()):
            st.markdown(f"### Section {sec}")

            work_section = work[work["Section"] == sec]

            dot = cached_performance_dfg(
                work_section,
                col_case,
                col_act,
                col_time
            )

            st.graphviz_chart(dot, width='stretch')

            st.markdown("---")

    else:
        pass
    st.markdown(f"### Full Process")
    # Default behavior (no Section column)
    st.write(
        "Below is the performance directly-follows graph (DFG) of the full dataset."
    )

    #dot = cached_performance_dfg(work, col_case, col_act, col_time)
    #st.graphviz_chart(dot, width='stretch')

def compute_dfg_counts(df_, case_col, act_col, time_col):
    """
    Returns:
      - dfg: dict of ((a,b) -> frequency)
      - activities: set of activities observed
      - start_acts: dict of (act -> count)
      - end_acts: dict of (act -> count)
    """
    dfg = {}
    activities = set()
    start_acts = {}
    end_acts = {}

    for _, g in df_.groupby(case_col, sort=False):
        g = g.sort_values(time_col)
        acts = g[act_col].tolist()
        if not acts:
            continue
        # starts/ends
        start_acts[acts[0]] = start_acts.get(acts[0], 0) + 1
        end_acts[acts[-1]] = end_acts.get(acts[-1], 0) + 1

        # activities
        for a in acts:
            activities.add(a)

        # edges
        for i in range(len(acts) - 1):
            key = (acts[i], acts[i+1])
            dfg[key] = dfg.get(key, 0) + 1

    return dfg, activities, start_acts, end_acts

@st.cache_data
def cached_event_log(work, case_col, act_col, time_col):
    return to_event_log(work, case_col, act_col, time_col)

def to_event_log(df_, case_col, act_col, time_col, res_col=None):
    """
    Robust conversion of a pandas DataFrame to a pm4py EventLog across pm4py versions.
    """
    from pm4py.objects.conversion.log import converter as log_converter

    keep = [case_col, act_col, time_col] + ([res_col] if res_col and res_col in df_.columns else [])
    tmp = df_[keep].copy()

    rename_map = {
        case_col: "case:concept:name",
        act_col: "concept:name",
        time_col: "time:timestamp",
    }
    if res_col and res_col in tmp.columns:
        rename_map[res_col] = "org:resource"

    tmp.rename(columns=rename_map, inplace=True)
    tmp["time:timestamp"] = pd.to_datetime(tmp["time:timestamp"], errors="coerce")
    tmp = tmp[tmp["time:timestamp"].notna()].copy()

    try:
        return log_converter.apply(tmp, variant=log_converter.Variants.TO_EVENT_LOG)
    except Exception:
        pass

    try:
        Params = log_converter.Variants.TO_EVENT_LOG.value.Parameters
        params = {}
        if hasattr(Params, "CASE_ID_KEY"):
            params[Params.CASE_ID_KEY] = "case:concept:name"
        if hasattr(Params, "ACTIVITY_KEY"):
            params[Params.ACTIVITY_KEY] = "concept:name"
        if hasattr(Params, "TIMESTAMP_KEY"):
            params[Params.TIMESTAMP_KEY] = "time:timestamp"
        if hasattr(Params, "RESOURCE_KEY") and "org:resource" in tmp.columns:
            params[Params.RESOURCE_KEY] = "org:resource"

        return log_converter.apply(tmp, variant=log_converter.Variants.TO_EVENT_LOG, parameters=params)
    except Exception:
        pass

    from pm4py.objects.conversion.log import factory as log_conv_factory
    try:
        return log_conv_factory.apply(tmp)
    except Exception as e:
        raise RuntimeError("Failed to convert DataFrame to EventLog.") from e

# ---------------------------
# Process model discovery with pm4py (Petri net) + Manual builder
# ---------------------------

net = im = fm = None
miner_name = None
pnml_file = None

if model_source == "Load stored PNML":
    if not PM4PY_AVAILABLE:
        st.error("pm4py is required to load a PNML Petri net. Install with `pip install pm4py`.")
        st.stop()

    # PNML path
    pnml_file = st.file_uploader("📥 Upload PNML", type=["pnml"])
    if pnml_file is not None:
        try:
            net, im, fm = import_petri_from_pnml_bytes(pnml_file.read())
            st.success("PNML model loaded.")
        except Exception as e:
            st.error(f"Failed to load PNML: {e}")

    # Fallback pickle ZIP path (optional)
    pickle_zip = st.file_uploader("📥 Or upload Pickle ZIP fallback", type=["zip"])
    if pickle_zip is not None:
        try:
            net, im, fm = import_petri_from_pickle_zip_bytes(pickle_zip.read())
            st.success("Pickle ZIP model loaded.")
        except Exception as e:
            st.error(f"Failed to load Pickle ZIP: {e}")

def convert_dfg_to_petri(dfg, start_acts, end_acts):
    """
    Version-safe DFG → Petri net conversion.
    """
    # pm4py >= 2.7 (most modern)
    try:
        from pm4py.algo.discovery.dfg import algorithm as dfg_algo
        return dfg_algo.apply_dfg(dfg, start_acts, end_acts)
    except:
        pass

    # pm4py 2.2–2.6
    try:
        from pm4py.objects.conversion.dfg import converter as dfg_converter
        return dfg_converter.apply(dfg, parameters={
            "start_activities": start_acts,
            "end_activities": end_acts
        })
    except:
        pass

    # pm4py < 2.2
    try:
        from pm4py.objects.conversion.dfg import factory as dfg_factory
        return dfg_factory.apply(dfg, start_acts, end_acts)
    except:
        pass

    raise RuntimeError("No compatible DFG → Petri conversion API available in your pm4py install.")

# ---------------------------
# Process model discovery with pm4py (Petri net) + Manual builder
# ---------------------------
if show_petri:
    st.subheader("🧭 Discovered / Loaded Process Model (Petri net)")

    if model_source == "Discover from event log":
        if not PM4PY_AVAILABLE and miner != "Manual (select flows)":
            st.warning(
                "pm4py is not available. Install it with `pip install pm4py` to discover and render Petri nets.\n"
                "You can still use the DFG above."
            )

        try:
            # Prepare an event log if pm4py is available
            log = None
            if PM4PY_AVAILABLE:
                log = cached_event_log(work, col_case, col_act, col_time)

            if miner and miner.startswith("Inductive"):
                if not PM4PY_AVAILABLE:
                    st.error("pm4py is required for Inductive Miner.")
                    st.stop()
                from pm4py import discover_petri_net_inductive
                from pm4py.algo.discovery.inductive import algorithm as inductive_miner
                tree = inductive_miner.apply(log)
                net, im, fm = discover_petri_net_inductive(log)
                miner_name = "Inductive Miner"

            elif miner and miner.startswith("Heuristics"):
                if not PM4PY_AVAILABLE:
                    st.error("pm4py is required for Heuristics Miner.")
                    st.stop()
                from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
                heu_net = heuristics_miner.apply_heu(log, parameters={"dependency_thresh": hm_threshold})
                net, im, fm = heuristics_miner.apply(log, parameters={"dependency_thresh": hm_threshold})
                miner_name = f"Heuristics Miner (dep={hm_threshold})"

            elif miner == "Manual (select flows)":
                # --- Manual builder (same as your original; trimmed for brevity, keep your full logic) ---
                st.caption("Manual model: pick which observed transitions should be included in the model, and which activities may serve as starting/ending activities. ")
                dfg_all, activities_all, start_all, end_all = compute_dfg_counts(work, col_case, col_act, col_time)
                if not dfg_all:
                    st.warning("No directly-follows relations found in the data.")
                    st.stop()

                edges_sorted = sorted(dfg_all.items(), key=lambda x: x[1], reverse=True)

                with st.expander("Manual Model Builder", expanded=True):
                    max_freq = max(freq for _, freq in edges_sorted)
                    min_freq = st.slider(
                        "Minimum edge frequency to show", 1, int(max_freq), min(3, int(max_freq)),
                        help="Only edges with frequency ≥ this value are shown in the selector."
                    )
                    candidate_edges = [(a, b, f) for (a, b), f in edges_sorted if f >= min_freq]
                    edge_labels = [f"{a} → {b}  (freq={f})" for a, b, f in candidate_edges]
                    default_sel = edge_labels[: min(15, len(edge_labels))]

                    selected_labels = st.multiselect(
                        "Select edges to include in the model",
                        options=edge_labels,
                        default=default_sel,
                    )

                    selected_dfg = {}
                    selected_activities = set()
                    for label, (a, b, f) in zip(edge_labels, candidate_edges):
                        if label in selected_labels:
                            selected_dfg[(a, b)] = f
                            selected_activities.update([a, b])

                    default_starts = [a for a in start_all.keys() if a in selected_activities] or list(start_all.keys())
                    default_ends = [a for a in end_all.keys() if a in selected_activities] or list(end_all.keys())

                    start_pick = st.multiselect(
                        "Start activities",
                        options=sorted(activities_all),
                        default=sorted(default_starts),
                    )
                    end_pick = st.multiselect(
                        "End activities",
                        options=sorted(activities_all),
                        default=sorted(default_ends),
                    )

                    start_acts = {a: start_all.get(a, 1) for a in start_pick}
                    end_acts = {a: end_all.get(a, 1) for a in end_pick}

                    st.markdown("**Preview of selected flows**")
                    if selected_dfg:
                        preview_lines = [f"- {a} → {b} (freq={f})" for (a, b), f in sorted(selected_dfg.items(), key=lambda x: x[1], reverse=True)]
                        st.write("\n".join(preview_lines))
                    else:
                        st.info("No edges selected yet.")

                    if selected_dfg:
                        if PM4PY_AVAILABLE:
                            try:
                                net, im, fm = convert_dfg_to_petri(selected_dfg, start_acts, end_acts)
                                miner_name = "Manual (selected flows)"
                            except Exception as e:
                                st.error(f"DFG → Petri conversion failed: {e}")
                        else:
                            st.warning("pm4py not installed — cannot build Petri net. You can still review the selected DFG above.")

            # Render if available (either discovered or manual)
            if net and im and fm:
                st.caption(f"Model Source: **{miner_name or 'Unknown'}**")
                gviz = pn_visualizer.apply(net, im, fm)
                try:
                    png_bytes = gviz.pipe(format="png")
                    st.image(png_bytes, caption="Petri net", width='stretch')
                except Exception:
                    st.graphviz_chart(gviz.source, width='stretch')

                # --- Download PNML ---
                # Try PNML first
                #try:
                #    pnml_bytes = export_petri_to_pnml_bytes(net, im, fm)
                #    st.download_button(
                #        "💾 Download model as PNML",
                #        data=pnml_bytes,
                #        file_name="process_model.pnml",
                #        mime="application/xml"
                #    )
                #except Exception as e:
                #    st.warning(f"PNML export unavailable on this pm4py build: {e}")
                #    # Offer Pickle ZIP as a fallback
                #    try:
                #        zip_bytes = export_petri_to_pickle_zip_bytes(net, im, fm)
                #        st.download_button(
                #            "💾 Download model (Pickle ZIP, fallback)",
                #            data=zip_bytes,
                #            file_name="process_model_pickle.zip",
                #            mime="application/zip"
                #        )
                #    except Exception as e2:
                #        st.error(f"Could not export model in any format: {e2}")

            else:
                if miner == "Manual (select flows)":
                    st.info("Choose flows and click **Build Petri Net** to construct the model.")

        except Exception as e:
            st.error(f"Discovery/visualization failed: {e}")
            st.info(
                "Common fixes:\n"
                "• Ensure timestamps are parsed correctly.\n"
                "• Try a different miner or adjust manual selections.\n"
                "• Install Graphviz and add it to PATH for better rendering."
            )

    else:
        # Model source: Loaded PNML
        if net and im and fm:
            st.caption(f"Model source: **{miner_name}**")
            try:
                gviz = pn_visualizer.apply(net, im, fm)
                try:
                    png_bytes = gviz.pipe(format="png")
                    st.image(png_bytes, caption="Petri net", width='stretch')
                except Exception:
                    st.graphviz_chart(gviz.source, width='stretch')
            except Exception as e:
                st.error(f"Failed to visualize loaded Petri net: {e}")

# ---------------------------
# Conformance Checking (TBR or Alignments)
# ---------------------------
if PM4PY_AVAILABLE:
    st.subheader("✔️ Conformance Checking")
    st.write(
        "This section provides details on how well the data conforms to the defined model. "
        "Conformance is commonly measured by **fitness**. Fitness is measured per case, providing a measure of how well a certain case fits the defined model. "
        "For overall model conformance, we calculate the mean fitness over all cases. "
        "A low fitness (with 0 being the lowest) indicates that the case (or dataset as a whole) does not really conform to the defined model. "
        "A high fitness (with 1 being the highest) indicates that the case (or the dataset as a whole) conforms relatively well to the defined model. "
        "A fitness of 1 indicates perfect conformance: the defined model accurately describes the observed process (which might indicate overfitting). "
    )

    # Ensure we have a Petri net and a pm4py log
    if 'net' not in locals() or net is None or im is None or fm is None:
        st.info("No Petri net available yet. Discover or load a model above.")
    else:
        try:
            # Build log now if not already built
            try:
                log = log # noqa: F821
            except NameError:
                log = cached_event_log(work, col_case, col_act, col_time)

            fitness_df = None

            if fitness_metric.startswith("Token-Based"):
                # -------- Token-Based Replay --------
                from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

                replay_result = token_replay.apply(
                    log, net, im, fm,
                    variant=token_replay.Variants.TOKEN_REPLAY
                )

                # Convert replay results → DataFrame
                fitness_rows = []
                for i, case in enumerate(replay_result):
                    fitness_rows.append({
                        "case_id": list(log)[i].attributes.get("concept:name", f"case_{i}"),
                        "fitness": case.get("trace_fitness", None),
                        "consumed": case.get("consumed_tokens", None),
                        "produced": case.get("produced_tokens", None),
                        "missing": case.get("missing_tokens", None),
                        "remaining": case.get("remaining_tokens", None),
                        "method": "TBR",
                    })
                fitness_df = pd.DataFrame(fitness_rows)

            else:
                # -------- Alignment-Based Conformance (version-safe) --------
                # Try multiple API locations across pm4py versions
                alignments = None
                align_factory = None
                try:
                    # Newer pm4py (most common)
                    from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
                except Exception:
                    try:
                        # Mid-generation pm4py
                        from pm4py.algo.conformance.alignments import algorithm as alignments
                    except Exception:
                        try:
                            # Very old fallback (factory)
                            from pm4py.algo.conformance.alignments import factory as align_factory
                        except Exception as _e:
                            raise ImportError(
                                "No usable alignment module found in this pm4py build."
                            )

                # Choose a robust default variant if available (depends on version)
                variant = None
                if alignments is not None and hasattr(alignments, "Variants"):
                    # Try several commonly present names across versions
                    for name in [
                        "VERSION_STATE_EQUATION_A_STAR",
                        "A_STAR",                     # some versions
                        "VERSION_TWEAKED",            # fallback name in a few builds
                        "DIJKSTRA"                    # least preferred, but present in some
                    ]:
                        if hasattr(alignments.Variants, name):
                            variant = getattr(alignments.Variants, name)
                            break

                # Run alignments using whatever API is available
                if alignments is not None:
                    if variant is not None:
                        alignment_result = alignments.apply_log(log, net, im, fm, variant=variant)
                    else:
                        # Try without explicit variant if we couldn’t resolve one
                        alignment_result = alignments.apply_log(log, net, im, fm)
                else:
                    # Factory fallback (very old pm4py)
                    alignment_result = align_factory.apply(log, net, im, fm)

                # Compute per-trace fitness in a version-safe way
                fitness_rows = []
                try:
                    # Preferred: dedicated evaluation util for alignment-based fitness
                    from pm4py.algo.evaluation.replay_fitness import algorithm as eval_fitness
                    eval_res = eval_fitness.evaluate(
                        log=log,
                        alignments=alignment_result,
                        variant=getattr(eval_fitness.Variants, "ALIGNMENT_BASED", None)
                                or getattr(eval_fitness.Variants, "ALIGNMENT_BASED_LOG", None)
                                or eval_fitness.Variants  # last resort fallback
                    )
                    per_trace_fitness = eval_res.get("per_trace", [])
                    for i, tr in enumerate(per_trace_fitness):
                        fitness_rows.append({
                            "case_id": list(log)[i].attributes.get("concept:name", f"case_{i}"),
                            "fitness": tr.get("fitness", None),
                            "cost": tr.get("cost", None),
                            "method": "Alignments",
                        })
                except Exception:
                    # Fallback: extract what we can from the raw alignments output
                    for i, ali in enumerate(alignment_result):
                        # Common keys: 'fitness', 'cost'; structure may vary
                        fit = None
                        cost = None
                        if isinstance(ali, dict):
                            fit = ali.get("fitness", None)
                            cost = ali.get("cost", None)
                        fitness_rows.append({
                            "case_id": list(log)[i].attributes.get("concept:name", f"case_{i}"),
                            "fitness": fit,
                            "cost": cost,
                            "method": "Alignments",
                        })

                fitness_df = pd.DataFrame(fitness_rows)

            # ----- Summaries (common to both methods) -----
            if fitness_df is not None and not fitness_df.empty and fitness_df["fitness"].notna().any():
                avg_fitness = fitness_df["fitness"].mean()
                median_fitness = fitness_df["fitness"].median()
                min_fitness = fitness_df["fitness"].min()

                st.metric("Average fitness", f"{avg_fitness:.3f}")
                st.write(
                    f"**Median fitness:** {median_fitness:.3f} | "
                    f"**Worst case fitness:** {min_fitness:.3f}"
                )

                with st.expander("Per-case conformance results"):
                    st.dataframe(fitness_df, width='stretch')

                st.subheader("📈 Fitness Distribution")
                st.write(
                    "The graph below visualizes the fitnesses of cases in the data. "
                )
                # Sort by fitness (descending), then plot
                sorted_df = fitness_df.sort_values('fitness', ascending=True).reset_index()
                st.bar_chart(data = sorted_df[['case_id', 'fitness']], x = None, y = 'fitness')

                # Histogram of fitness
                hist = (
                    alt.Chart(fitness_df.dropna(subset=["fitness"]))
                    .mark_bar(color="#4F46E5", opacity=0.7)
                    .encode(
                        x=alt.X("fitness:Q", bin=alt.Bin(maxbins=30), title="Fitness"),
                        y=alt.Y("count()", title="Cases"),
                        tooltip=[alt.Tooltip("count()", title="Cases")]
                    )
                )
                st.subheader("📊 Fitness Distribution")
                st.altair_chart(hist, width='stretch')

                # KPI thresholds
                threshold = st.slider("Fitness pass threshold", 0.0, 1.0, 0.9, 0.01)
                passed = (fitness_df["fitness"] >= threshold).sum()
                total  = len(fitness_df)
                st.metric("Cases meeting threshold", f"{passed}/{total} ({passed/total:.0%})")
            else:
                st.info("No fitness values computed. Check your model/log or try the other method.")

        except Exception as e:
            st.error(f"Conformance checking failed: {e}")
            st.info(
                "If this error persists, ensure that the discovered Petri net is sound "
                "and that pm4py is updated."
            )

else:
    st.info("Install pm4py to enable conformance checking.")

# ---------------------------
# Extra EDA (after Fitness Distribution)
# ---------------------------
st.markdown("---")
st.header("🔬 Some Analytical Results")

# ==============
# 1) Case-level analytics
# ==============
st.subheader("📦 Case-level Analytics")

# Case length (number of events per case)
@st.cache_data
def cached_case_lengths(df_, case_col, act_col):
    return df_.groupby(case_col)[act_col].size().rename("events_per_case").reset_index()

@st.cache_data
def cached_case_durations(df_, case_col, time_col):
    agg = df_.groupby(case_col).agg(start=(time_col, "min"), end=(time_col, "max"))
    agg["throughput_seconds"] = (agg["end"] - agg["start"]).dt.total_seconds()
    return agg

case_lengths = cached_case_lengths(work, col_case, col_act)
case_durs = cached_case_durations(work, col_case, col_time)

# Case throughput times
def compute_case_durations(df_, case_col, time_col):
    agg = df_.groupby(case_col).agg(start=(time_col, "min"), end=(time_col, "max"))
    agg["throughput_seconds"] = (agg["end"] - agg["start"]).dt.total_seconds()
    return agg

case_durs = compute_case_durations(work, col_case, col_time)

# Merge with durations (already computed earlier as case_durs)
case_summary = case_lengths.merge(
    case_durs.reset_index()[[col_case, "throughput_seconds"]],
    on=col_case,
    how="left"
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Avg events per case", f"{case_summary['events_per_case'].mean():.2f}")
with col2:
    if case_summary['throughput_seconds'].notna().any():
        st.metric("Avg throughput (hh:mm:ss)",
                  str(timedelta(seconds=int(np.nanmean(case_summary['throughput_seconds'])))))
    else:
        st.metric("Avg throughput (hh:mm:ss)", "n/a")
with col3:
    st.metric("Median events per case", f"{case_summary['events_per_case'].median():.0f}")

st.write("**Bar Chart: Events per case**")
st.write(
    "This graph visualizes the number of events the cases went through. "
)

activities_freq = pd.DataFrame(case_summary["events_per_case"].value_counts().sort_index())
activities_freq['events_per_case'] = activities_freq.index
bars = (
    alt.Chart(activities_freq)
    .mark_bar(color="#4F46E5")
    .encode(
        # Use the rank number on the x-axis
        x=alt.X("events_per_case:O", title="Number of Events that Took Place", sort=None),  # 'sort=None' preserves DataFrame order
        y=alt.Y("count:Q", title="Number of Observations"),
        tooltip=[
            alt.Tooltip("events_per_case:Q", title="Number of Events that Took Place"),
            alt.Tooltip("count:N", title="Number of Observations"),
        ],
    )
)
st.altair_chart(bars, width='stretch')

# ==============
# 2) Variant analysis (Pareto)
# ==============
st.subheader("🧬 Variant Analysis (Pareto)")

@st.cache_data
def cached_variants(work, col_case, col_act):
    return work.groupby(col_case)[col_act].apply(tuple).value_counts().reset_index()

variants_df = cached_variants(work, col_case, col_act)
variants_df.columns = ["Variant", "Number of Cases"]
variants_df["Cumulative Number of Cases"] = variants_df["Number of Cases"].cumsum()
variants_df["Cumulative Percentage of Dataset"] = variants_df["Cumulative Number of Cases"] / variants_df["Number of Cases"].sum() * 100.0

n = len(variants_df[variants_df["Cumulative Percentage of Dataset"] <= 80.0])
st.write(f"**Top {n} variants**")
st.write("The table below shows which variants occurred most frequently in the given dataset, sorted by frequency. ")
st.dataframe(variants_df.head(n))

# Pareto chart (bars = cases, line = cumulative %)
st.write(
    "The graph below visualizes the number of times each variant was observed. "
)

top_n = min(n, len(variants_df))
vplot_df = variants_df.head(top_n).copy()
vplot_df["variant_no"] = range(0, len(vplot_df))
vplot_df["variant_str"] = vplot_df["Variant"].astype(str).str.slice(0, 80)
bars = (
    alt.Chart(vplot_df)
    .mark_bar(color="#4F46E5")
    .encode(
        # Use the rank number on the x-axis
        x=alt.X("variant_no:O", title="Variant #", sort=None),  # 'sort=None' preserves DataFrame order
        y=alt.Y("Number of Cases:Q", title="Number of Cases"),
        tooltip=[
            alt.Tooltip("variant_no:O", title="Variant #"),
            alt.Tooltip("Number of Cases:Q", title="Number of Cases"),
            alt.Tooltip("variant_str:N", title="Variant (truncated)"),
        ],
    )
)
st.altair_chart(bars, width='stretch')

# ==============
# 3) Activity-level analytics
# ==============
st.subheader("🏷️ Activity Analytics")

# Frequency (already computed as activity_freq earlier; recompute to be safe)
@st.cache_data
def cached_activity_freq(df_, act_col):
    freq = df_[act_col].value_counts().reset_index()
    freq.columns = ["Activity", "Events"]
    return freq

activity_freq = cached_activity_freq(work, col_act)
activity_freq.columns = ["Activity", "Events"]
st.write("**Top activities by frequency**")
st.dataframe(activity_freq.head(20))

# Approximate per-activity "service time":
# difference between consecutive timestamps within the same case for the *from* activity.
# This approximates how long we spend "between" A and next event B (waiting + service).
tmp = work[[col_case, col_act, col_time]].copy()
tmp["next_time"] = tmp.groupby(col_case)[col_time].shift(-1)
tmp["delta_sec"] = (tmp["next_time"] - tmp[col_time]).dt.total_seconds() / (24*3600)
serv = tmp.groupby(col_act)["delta_sec"].agg(['count', 'mean', 'median']).reset_index()
serv.rename(columns={"activity": "Activity", "count": "Number of Times the Activity was Completed", "mean": "Average Time to Next Activity (days)", "median": "Median Time to Next Activity (days)"}, inplace=True)

st.write("**Per-activity average time to next event (days)**")
st.dataframe(serv.sort_values("Average Time to Next Activity (days)", ascending=False).head(20).reset_index().drop(columns = ["index"]))

# ===========================
# NEW: Cases skipping each activity (model-aware)
# ===========================
st.write("---")
st.subheader("🚫 Skipped Activities")
st.write('This section looks at activities skipped by contracts in the ideal process.')

# -------------------------------------------------
# Filter option: only cases appearing in BOTH sections
# -------------------------------------------------

if "Section" in work.columns:
    only_common_cases = st.checkbox(
        "Include only Case IDs found in both sections",
        value=False,
        help="If checked, skip analysis is restricted to cases that appear in multiple sections (e.g. CLM and Coupa Core)."
    )
else:
    only_common_cases = False

if only_common_cases and "Section" in work.columns:
    eligible_cases = (
        work.groupby(col_case)["Section"]
            .nunique()
            .loc[lambda s: s >= 2]
            .index
    )
    work_for_skips = work[work[col_case].isin(eligible_cases)].copy()
else:
    work_for_skips = work

if only_common_cases:
    st.caption(
        f"🔎 Skip analysis restricted to {work_for_skips[col_case].nunique()} cases found in both sections."
    )

def _get_model_activities_if_available(net_obj, fallback_series):
    """
    If a Petri net is available, return the set of labeled transitions (i.e., model activities).
    Otherwise return observed activities from the event log.
    """
    try:
        if net_obj is not None:
            # Only keep visible transitions with non-empty labels
            acts = sorted({t.label for t in net_obj.transitions if getattr(t, "label", None)})
            if acts:
                return acts
    except Exception:
        pass
    # Fallback: observed activities from the dataset
    return sorted(fallback_series.dropna().unique().tolist())

@st.cache_data
def compute_skips_per_activity(df_, case_col, act_col, activities_universe):
    """
    For each activity in 'activities_universe', compute:
      - number of cases skipping that activity
      - which case IDs skip it
      - number of cases that do contain it
      - share (%) of cases that skip it
    Returns a summary DataFrame sorted by number of cases skipping (desc).
    """
    # All distinct cases
    all_cases = set(df_[case_col].dropna().unique())

    # Cases per observed activity
    cases_by_act = (
        df_.groupby(act_col)[case_col]
        .apply(lambda s: set(s.dropna().unique()))
        .to_dict()
    )

    rows = []
    total_cases = len(all_cases)
    for a in activities_universe:
        present = cases_by_act.get(a, set())
        missing = all_cases - present
        rows.append({
            "Activity": a,
            "Cases skipping": len(missing),
            "Share skipping (%)": (len(missing) / total_cases * 100.0) if total_cases else 0.0,
            "Cases present": len(present),
            "Total cases": total_cases,
            "Missing case IDs": sorted(list(missing)),
        })
    out = pd.DataFrame(rows).sort_values("Cases skipping", ascending=False, kind="mergesort")
    return out

# Determine which activity universe to use (model-aware if a Petri net is available)
try:
    model_activities = _get_model_activities_if_available(net if 'net' in locals() else None, work[col_act])
except Exception:
    model_activities = sorted(work[col_act].dropna().unique().tolist())

# -------------------------------------------------
# Determine which cases are eligible for skip analysis
# -------------------------------------------------

skip_df = compute_skips_per_activity(
    work_for_skips,
    col_case,
    col_act,
    model_activities
)

# Top-N selector for chart readability
top_k = st.slider("Number of top skipped activities to show:", 5, max(5, len(skip_df)), min(15, len(skip_df)))

# --- Bar chart: number of cases skipping each activity ---
chart_df = skip_df[["Activity", "Cases skipping"]].head(top_k).copy()
bars = (
    alt.Chart(chart_df)
    .mark_bar(color="#EF4444")
    .encode(
        y=alt.Y("Activity:N", sort="-x", title="Activity"),
        x=alt.X("Cases skipping:Q", title="Number of Skips"),
        tooltip=[
            alt.Tooltip("Activity:N"),
            alt.Tooltip("Number of Skips:Q", title="# Cases skipping"),
        ],
    )
)
st.altair_chart(bars, width='stretch')

# --- Summary table + expanders for case lists ---
st.write("**Summary: Skipped Activities**")

skip_df2 = skip_df[["Activity", "Cases skipping", "Share skipping (%)", "Cases present", "Total cases"]].rename(
    columns={"Total cases": "Total Number of Contracts", "Cases skipping": "Total Number of Skips", "Share skipping (%)": "Percentage out of Total Number of Contracts", "Cases present": "Total Number of Contracts Having Executed Activity"}
)
st.dataframe(
    skip_df2,
    width='stretch'
)

with st.expander("🔎 Explore cases skipping each activity"):
    CASES_PER_GROUP = 100  # max cases per subgroup (controls granularity)

    for _, r in skip_df.iterrows():
        activity = r["Activity"]
        missing_cases = r["Missing case IDs"]

        # Clean + normalize case IDs
        missing_cases = sorted(int(c) for c in missing_cases)
        missing_cases_str = [str(c) for c in missing_cases]

        # -------- Activity-level expander (KEY FIX #1) --------
        with st.expander(
            f"🏷️ {activity} — {len(missing_cases)} case(s) skipping",
            expanded=False
        ):
            if not missing_cases:
                st.caption("✅ All cases include this activity.")
                continue

            # -------- Split by CASE-ID ranges (KEY FIX #2) --------
            for i in range(0, len(missing_cases), CASES_PER_GROUP):
                chunk_ids = missing_cases[i:i + CASES_PER_GROUP]
                chunk_str = [str(c) for c in chunk_ids]

                range_label = f"Case IDs [{chunk_ids[0]} – {chunk_ids[-1]}]"

                with st.expander(range_label, expanded=False):
                    st.write(", ".join(chunk_str))

# Optional: Download long-form CSV (one row per activity–missing-case)
long_form = skip_df[["Activity", "Missing case IDs"]].explode("Missing case IDs").rename(
    columns={"Missing case IDs": "Case ID"}
)
csv_bytes = long_form.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download long-form CSV of (Activity, Case ID) pairs for skipped activities",
    data=csv_bytes,
    file_name="cases_skipping_activities.csv",
    mime="text/csv",
)

# ==============
# 4) Transition analytics (waiting-time per directly-follows)
# ==============
st.subheader("➡️ Transition (A → B) Analytics")

@st.cache_data
def cached_transition_pairs(df_, case_col, act_col, time_col):
    pairs = []
    for case_id, g in df_.groupby(case_col, sort=False):
        g = g.sort_values(time_col)
        acts = g[act_col].tolist()
        times = g[time_col].tolist()
        for i in range(len(acts) - 1):
            dt = (times[i+1] - times[i]).total_seconds()
            if pd.notnull(dt):
                pairs.append((case_id, acts[i], acts[i+1], dt))
    return pd.DataFrame(pairs, columns=["case_id", "from_act", "to_act", "delta_sec"])

pairs_df = cached_transition_pairs(work, col_case, col_act, col_time)

if pairs_df is not None:
    @st.cache_data
    def cached_transition_stats(pairs_df):
        return pairs_df.groupby(["from_act", "to_act"])["delta_sec"].agg(
            count="count",
            avg="mean",
            p50="median",
            p90=lambda x: np.percentile(x, 90),
            p95=lambda x: np.percentile(x, 95)
        ).reset_index()

    agg = cached_transition_stats(pairs_df)
    st.write("**Slowest transitions by average time**")
    df_showing = agg.copy()
    df_showing["avg"] = df_showing["avg"] / (24*3600)
    df_showing = df_showing.drop(columns = ["p50", "p90", "p95"])
    df_showing.rename(columns={"from_act": "'From' Activity", "to_act": "'To' Activity", "count": "Count", "avg": "Average Transition Time (days)"}, inplace=True)
    st.dataframe(df_showing.head(20).reset_index().drop(columns = ["index"]))

    # ---- Histograms per transition (hours) ----

    st.write("⏳ **Transition Duration Histograms (days)**")
    st.caption("Select a transition to view its duration distribution.")

    # Build a selection list
    transition_list = (
        pairs_df[["from_act", "to_act"]]
        .drop_duplicates()
        .apply(lambda row: f"{row['from_act']} → {row['to_act']}", axis=1)
        .tolist()
    )

    selected_transition = st.selectbox(
        "Choose a transition:",
        transition_list,
    )

    if selected_transition:
        a, b = selected_transition.split(" → ")
        hist_df = pairs_df[(pairs_df["from_act"] == a) & (pairs_df["to_act"] == b)].copy()
        hist_df["days"] = hist_df["delta_sec"] / (24*3600.0)

        st.write(f"#### Distribution for **{a} → {b}** (days)")

        # ------------------------------------
        # KDE computation (Python-side)
        # ------------------------------------
        if len(hist_df) > 1:
            from sklearn.neighbors import KernelDensity

            x = hist_df["days"].values.reshape(-1, 1)

            # KDE bandwidth heuristic
            bandwidth = max(0.1, np.std(x) * 0.3)

            kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(x)

            # Generate a smooth range of x-values
            x_min, x_max = x.min(), x.max()
            xs = np.linspace(x_min, x_max, 200).reshape(-1, 1)

            log_density = kde.score_samples(xs)
            density = np.exp(log_density)

            kde_df = pd.DataFrame({
                "days": xs.flatten(),
                "density": density
            })
            density_scaled = density * len(hist_df) * ( (x_max - x_min) / 40 )
            kde_df["density"] = density_scaled
        else:
            kde_df = None

        # ------------------------------------
        # Histogram + KDE (Altair)
        # ------------------------------------

        normal_hist = (
            alt.Chart(hist_df)
            .mark_bar(opacity=0.6, color="#4F46E5")
            .encode(
                x=alt.X("days:Q", bin=alt.Bin(maxbins=40), title="Duration (days)"),
                y=alt.Y("count()", title="Frequency"),
            )
        )

        if kde_df is not None:
            kde_line = (
                alt.Chart(kde_df)
                .mark_line(color="#DC2626", size=2)
                .encode(
                    x="days:Q",
                    y=alt.Y("density:Q", axis=alt.Axis(title="Density")),
                )
            )
            chart = normal_hist + kde_line
        else:
            chart = None

        if chart is not None:
            st.altair_chart(chart, width='stretch')

        # ------------------------------------
        # Histogram layer (Altair)
        # ------------------------------------

        def split_histogram_brokenaxes(
            hist_df,
            value_col="days",
            maxbins=40,
            # --- Auto mode tuning (used if abstract_range is None) ---
            iqr_k=1.5,               # Tukey outlier fence multiplier
            bottom_quantile=0.97,    # keep bottom panel capped ~97th pct of bin counts
            # --- Manual mode (user-selected gap[s]) ---
            abstract_range=None,     # None | (y_low, y_high) | [(y_low, y_high), ...]
            # --- Shared styling ---
            headroom=0.01,           # headroom above the tallest bin in top panel
            bar_color="#4F46E5",
            bar_opacity=0.8,
            figsize=(8, 6),
            hspace=0.1              # small gap between panels
        ):
            """
            Broken-axis histogram. Two modes:

            1) AUTO (default): Detects outlier bars (by count) and picks a split so that
            tiny bars remain visible in the bottom panel.

            2) MANUAL (if `abstract_range` is provided): You specify the y-axis range(s)
            of bar heights (frequencies) to hide, and the histogram is drawn with
            corresponding broken y-axes.

            Parameters
            ----------
            hist_df : pandas.DataFrame
                Input data frame containing the numeric column for the histogram.
            value_col : str
                Column name with numeric values to histogram.
            maxbins : int
                Maximum number of bins for `np.histogram`.
            iqr_k : float
                Tukey IQR multiplier for the upper outlier fence (AUTO mode only).
            bottom_quantile : float in (0, 1)
                Upper cap for the bottom panel in AUTO mode (as a high quantile of
                non-zero bin counts).
            abstract_range : None | tuple | list of tuples
                If None -> AUTO mode.
                If (y_low, y_high) -> MANUAL mode: hide this frequency band.
                If list of tuples -> MANUAL mode: hide multiple frequency bands.
            headroom : float
                Fractional headroom above the tallest bar for the top panel.
            bar_color : str
                Bar color.
            bar_opacity : float
                Bar alpha.
            figsize : (float, float)
                Figure size (inches).
            hspace : float
                Vertical spacing between broken panels.

            Returns
            -------
            fig : matplotlib.figure.Figure
            """

            # 1) Compute histogram bins and counts
            values = hist_df[value_col].dropna().to_numpy()
            if values.size == 0:
                fig, ax = plt.subplots(figsize=figsize)
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.axis("off")
                return fig

            counts, edges = np.histogram(values, bins=maxbins)
            widths = np.diff(edges)
            lefts  = edges[:-1]

            if counts.max() == 0:
                fig, ax = plt.subplots(figsize=figsize)
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.axis("off")
                return fig

            max_count = float(counts.max())
            top_max = max_count * (1 + headroom)

            # Helper: normalize abstract_range into a sorted, non-overlapping list of tuples
            def _normalize_ranges(ranges, ymax):
                # Convert to list of tuples
                if isinstance(ranges, tuple) and len(ranges) == 2:
                    ranges = [ranges]
                elif not isinstance(ranges, (list, tuple)):
                    raise ValueError("abstract_range must be a (y_low, y_high) tuple or a list of such tuples.")

                # Clean and clamp each range
                cleaned = []
                for r in ranges:
                    if not (isinstance(r, (list, tuple)) and len(r) == 2):
                        raise ValueError("Each abstract range must be a (y_low, y_high) pair.")
                    lo, hi = float(r[0]), float(r[1])
                    # Swap if reversed
                    if lo > hi:
                        lo, hi = hi, lo
                    # Clamp to [0, ymax]
                    lo = max(0.0, min(lo, ymax))
                    hi = max(0.0, min(hi, ymax))
                    # Keep only if there's a positive gap
                    if hi - lo > 1e-9:
                        cleaned.append((lo, hi))

                if not cleaned:
                    return []

                # Merge overlapping/adjacent ranges
                cleaned.sort()
                merged = [cleaned[0]]
                for lo, hi in cleaned[1:]:
                    prev_lo, prev_hi = merged[-1]
                    if lo <= prev_hi + 1e-9:  # overlap or touch
                        merged[-1] = (prev_lo, max(prev_hi, hi))
                    else:
                        merged.append((lo, hi))
                return merged

            # 2) Build y-lims segments (panels) for brokenaxes
            #    In MANUAL mode: panels = stretches outside the hidden ranges.
            #    In AUTO mode: panels = two stretches separated by the auto-chosen split.
            def _build_manual_panels(hidden_ranges, ymax, headroom_factor=1.0):
                """
                Given a list of hidden ranges [(lo, hi), ...], return ylims segments
                that cover [0, ymax] excluding those hidden intervals.
                """
                segments = []
                cursor = 0.0
                for lo, hi in hidden_ranges:
                    if cursor < lo:
                        segments.append((cursor, lo))
                    cursor = max(cursor, hi)
                if cursor < ymax:
                    segments.append((cursor, ymax * headroom_factor))
                # Ensure each segment has positive height
                segments = [(a, b) for (a, b) in segments if b - a > 1e-9]
                return tuple(segments)

            # AUTO mode (original logic) unless abstract_range is provided
            if abstract_range is None:
                # 2) Outlier detection on non-zero counts
                nz = counts[counts > 0]
                q1, q3 = np.quantile(nz, [0.25, 0.75])
                iqr = max(q3 - q1, 0.0)
                high_fence = q3 + iqr_k * iqr
                outliers_mask = counts > high_fence
                outliers = counts[outliers_mask]

                # If no outliers → normal histogram (no break)
                if outliers.size == 0:
                    fig, ax = plt.subplots(figsize=figsize)
                    ax.bar(lefts, counts, width=widths, align="edge",
                        color=bar_color, alpha=bar_opacity)
                    ax.set_xlabel(value_col)
                    ax.set_ylabel("Frequency")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    return fig

                # --- NEW: two-limits approach for better bottom-panel readability ---
                normal = nz[nz <= high_fence]
                smallest_outlier = float(outliers.min())
                tallest_normal   = float(normal.max())

                # Place gap between tallest normal and smallest extreme
                y_lower_max = max(1.0, 1.05 * tallest_normal)
                y_upper_min = max(1.0, 0.95 * smallest_outlier)

                # Safety: distinct and within bounds
                if y_upper_min <= y_lower_max:
                    mid = 0.5 * (y_lower_max + y_upper_min)
                    y_lower_max = max(1.0, mid - 1.0)
                    y_upper_min = mid + 1.0

                y_lower_max = min(y_lower_max, max_count * 0.98)
                y_upper_min = min(y_upper_min, max_count * 0.99)

                # 4) Build the broken-axes figure with two panels
                fig = plt.figure(figsize=figsize)
                bax = brokenaxes(
                    ylims=((0, y_lower_max), (y_upper_min, top_max)),
                    hspace=hspace,
                    despine=True,
                    figure=fig
                )

            else:
                # MANUAL mode: user-selected gap(s)
                hidden = _normalize_ranges(abstract_range, max_count)
                if not hidden:
                    # Nothing to hide -> draw a normal histogram
                    fig, ax = plt.subplots(figsize=figsize)
                    ax.bar(lefts, counts, width=widths, align="edge",
                        color=bar_color, alpha=bar_opacity)
                    ax.set_xlabel(value_col)
                    ax.set_ylabel("Frequency")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    return fig
                
                # Create y segments outside the hidden ranges, last panel gets headroom
                panels = _build_manual_panels(hidden, max_count, headroom_factor=(1 + headroom))

                # If everything got hidden (shouldn't happen with the checks), fall back gracefully
                if len(panels) == 0:
                    fig, ax = plt.subplots(figsize=figsize)
                    ax.bar(lefts, counts, width=widths, align="edge",
                        color=bar_color, alpha=bar_opacity)
                    ax.set_xlabel(value_col)
                    ax.set_ylabel("Frequency")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    return fig

                fig = plt.figure(figsize=figsize)
                bax = brokenaxes(
                    ylims=panels,  # can be 2+ segments if multiple gaps are provided
                    hspace=hspace,
                    despine=True,
                    figure=fig
                )

            # 5) Draw bars on the broken axes
            bax.bar(lefts, counts, width=widths, align="edge",
                    color=bar_color, alpha=bar_opacity, edgecolor=bar_color)

            # Cosmetics
            bax.set_xlabel(value_col)
            bax.set_ylabel("Frequency")

            # Try to remove outer spines for a cleaner, balanced look
            try:
                for ax in fig.axes:
                    if hasattr(ax, "spines"):
                        ax.spines.get("top", None) and ax.spines["top"].set_visible(False)
                        ax.spines.get("right", None) and ax.spines["right"].set_visible(False)
            except Exception:
                pass

            return fig

        # --- 0) Compute data-driven default abstraction range from the current data ---

        # Pull values and build a histogram once (same params as the plotting function)
        values = hist_df["days"].dropna().to_numpy()
        counts, edges = np.histogram(values, bins=40)  # keep in sync with maxbins
        max_count = int(counts.max()) if counts.size else 0
        min_count = int(counts.min()) if counts.size else 0

        # --- 1) Streamlit inputs now use data-driven defaults ---

        abstraction_min = st.number_input(
            label="Enter the minimum value to be abstracted:",
            min_value=0,                 # Minimum allowed value
            max_value=int(max_count),    # Bind to actual data range
            value=int(max_count*0.05),
            step=1,                      # Integers
            format="%d"
        )

        abstraction_max = st.number_input(
            label="Enter the maximum value to be abstracted:",
            min_value=0,
            max_value=int(max_count),
            value=int(max_count*0.95),
            step=1,
            format="%d"
        )

        # Optional: guard that min < max by gently coercing (prevents user inversion)
        if abstraction_max <= abstraction_min:
            st.info("Adjusting abstract range to maintain a positive gap.")
            abstraction_max = min(int(max_count), abstraction_min + 2)

        # --- 2) Draw with your existing function in MANUAL mode ---

        broken_hist = split_histogram_brokenaxes(
            hist_df,
            value_col="days",
            maxbins=40,
            abstract_range=(abstraction_min, abstraction_max),  # your selected y-gap in frequency units
        )

        st.pyplot(broken_hist)

    st.write("**Most frequent transitions**")
    st.dataframe(df_showing.sort_values("Count", ascending=False).head(20).reset_index().drop(columns = ["index"]))

# ==============
# 5) WIP (Work-in-Progress) Over Time
# ==============
st.subheader("📈 Work-in-Progress (WIP) Over Time")
st.write(
    "The graph below shows how many cases were going through the process, over time. "
)

# Build case start/end from earlier case_durs
wip_df = case_durs.reset_index()[[col_case, "start", "end"]].dropna()
if not wip_df.empty:
    # Create +1 at start and -1 at end; cumulative sum is WIP
    stamps = []
    for _, row in wip_df.iterrows():
        stamps.append((row["start"], +1))
        # Small epsilon to drop WIP at exact end second (optional)
        stamps.append((row["end"], -1))
    wip_t = pd.DataFrame(stamps, columns=["timestamp", "delta"]).sort_values("timestamp")
    wip_t["wip"] = wip_t["delta"].cumsum()
    # Downsample if very dense
    if len(wip_t) > 5000:
        # sample 1 per N rows as a simple thinning
        step = max(1, len(wip_t)//2000)
        wip_t = wip_t.iloc[::step, :].copy()
    st.line_chart(wip_t.set_index("timestamp")["wip"])
else:
    st.info("Not enough data to compute WIP.")
