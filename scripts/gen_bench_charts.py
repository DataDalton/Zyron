#!/usr/bin/env python3
"""Regenerate the benchmark charts in README.md from the committed benchmark JSON.

For each suite in benchmarks/, the newest result file (by the timestamp in its
filename) is read. A curated set of metrics is pulled from those files and
rendered as Mermaid xychart-beta blocks plus a summary table, written between
the <!-- BENCH:START --> and <!-- BENCH:END --> markers in README.md.

Curation lives in throughput and latency below. The numbers themselves always
come from the latest benchmark files, so re-running this after a new benchmark
run refreshes the charts. Standard library only.

Usage:  python scripts/gen_bench_charts.py
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"
README = ROOT / "README.md"

# (suite, test group, metric key, short label, divisor)  -- value shown = average / divisor
THROUGHPUT = [
    ("storage", "B+ Tree", "Insert throughput (ops/sec)", "B+tree insert", 1e6),
    ("storage", "B+ Tree", "Delete throughput (ops/sec)", "B+tree delete", 1e6),
    ("executor", "Hash Build", "Hash build throughput (rows/sec)", "Hash join build", 1e6),
    ("versioning", "scd_type2_merge_throughput", "rows_per_sec", "SCD-2 merge", 1e6),
    ("wire", "COPY FROM", "COPY FROM (CSV) throughput (rows/sec)", "COPY FROM", 1e6),
    ("wire", "Row Serialization", "Row serialization throughput (rows/sec)", "Row serialize", 1e6),
    ("parser", "Parser Batch", "Throughput (stmts/sec)", "SQL parse", 1e6),
    ("lifecycle", "ttl_row_delete", "throughput (rows/sec)", "TTL purge", 1e6),
    ("lifecycle", "archive_to_object_store", "throughput (rows/sec)", "Archive", 1e6),
]

# (suite, test group, metric key, short label)  -- value shown in ns
LATENCY = [
    ("transaction", "Phase 1.5 Microbenchmarks", "is_visible() latency (ns/op)", "MVCC visible"),
    ("versioning", "version_log_lookup_by_id", "ns_per_lookup", "Version lookup"),
    ("storage", "B+ Tree", "Lookup latency (ns/op)", "B+tree lookup"),
    ("versioning", "branch_page_resolution", "ns_per_resolve", "Branch resolve"),
    ("transaction", "Phase 1.5 Microbenchmarks", "lock_row() latency (ns/op)", "Row lock"),
    ("lifecycle", "legal_hold_check_no_holds", "latency (ns/op)", "Legal-hold check"),
    ("optimizer", "optimizer", "selectivity_estimate_ns", "Selectivity est"),
]

# Supplementary table, rendered below the charts.
# (suite, group, metric key, subsystem, metric label, divisor, decimals, suffix)
EXTRA = [
    ("transaction", "Phase 1.5 Microbenchmarks", "GC sweep throughput (tuples/sec)", "MVCC", "GC sweep", 1e9, 1, "B tuples/sec"),
    ("columnar", "Parallel Column Encoding", "Sequential compaction (rows/sec)", "Columnar", "Sequential compaction", 1e6, 1, "M rows/sec"),
    ("versioning", "time_travel_scan_overhead", "overhead_percent", "Versioning", "Time-travel scan overhead", 1, 0, "%"),
    ("wire", "QUIC PG Handshake", "QUIC PG handshake latency (us)", "Wire", "QUIC PostgreSQL handshake", 1, 0, " us"),
]


def latest_json(suite: str):
    files = sorted((BENCH / suite).glob(f"{suite}_*.json"))
    if not files:
        raise SystemExit(f"no benchmark json for suite '{suite}'")
    raw = files[-1].read_text(encoding="utf-8")
    # Some result files emit bare `inf`/`nan` for unbounded targets; not valid JSON.
    raw = re.sub(r":\s*-?inf\b", ": 1e308", raw)
    raw = re.sub(r":\s*nan\b", ": null", raw)
    return json.loads(raw)


def metric(doc, group, key):
    return doc["tests"][group][key]["average"]


def bar_block(title, axis_label, labels, values, top):
    quoted = ", ".join(f'"{l}"' for l in labels)
    nums = ", ".join(f"{v:.1f}" for v in values)
    # Widen the canvas with bar count so x-axis labels do not overlap. The SVG
    # scales to the container, so a wider canvas just gives each label more room.
    width = max(720, 150 * len(labels))
    theme = (
        "%%{init: {'theme':'base','themeVariables':{'xyChart':{"
        "'backgroundColor':'transparent',"
        "'titleColor':'#e6edf3',"
        "'xAxisLabelColor':'#e6edf3','xAxisTitleColor':'#e6edf3',"
        "'xAxisTickColor':'#1f6feb','xAxisLineColor':'#1f6feb',"
        "'yAxisLabelColor':'#e6edf3','yAxisTitleColor':'#e6edf3',"
        "'yAxisTickColor':'#1f6feb','yAxisLineColor':'#1f6feb',"
        "'plotColorPalette':'#1f6feb'}},"
        f"'xyChart':{{'width':{width},'height':360}}}}}}%%"
    )
    return (
        "```mermaid\n"
        f"{theme}\n"
        "xychart-beta\n"
        f'    title "{title}"\n'
        f"    x-axis [{quoted}]\n"
        f'    y-axis "{axis_label}" 0 --> {top}\n'
        f"    bar [{nums}]\n"
        "```\n"
    )


def main():
    cache = {}

    def doc(suite):
        if suite not in cache:
            cache[suite] = latest_json(suite)
        return cache[suite]

    tp_labels, tp_vals = [], []
    for suite, group, key, label, div in THROUGHPUT:
        tp_labels.append(label)
        tp_vals.append(metric(doc(suite), group, key) / div)

    lat_labels, lat_vals = [], []
    for suite, group, key, label in LATENCY:
        lat_labels.append(label)
        lat_vals.append(metric(doc(suite), group, key))

    # End-to-end: a client talking to a running server over the wire.
    e2e = doc("end_to_end")
    cold = metric(e2e, "Startup", "cold boot latency (ms)")
    first_q = metric(e2e, "Bootstrap", "first ReadyForQuery (ms)")
    ddl = metric(e2e, "Bootstrap", "schema DDL total (ms)")
    seed = metric(e2e, "Bootstrap", "seed insert (rows/sec)")
    teardown = metric(e2e, "Shutdown", "teardown (ms)")
    analytics_us = metric(e2e, "Analytics", "median query us")
    conns = [("c1", "1 client"), ("c4", "4 clients"), ("c16", "16 clients")]
    tps = {c: metric(e2e, "OLTP", f"{c} tps") for c, _ in conns}
    p99 = {c: metric(e2e, "OLTP", f"{c} p99 us") for c, _ in conns}

    e2e_table = (
        "| Lifecycle / workload | Result |\n"
        "|----------------------|--------|\n"
        f"| Cold boot to accepting queries | {cold:.0f} ms |\n"
        f"| First `ReadyForQuery` | {first_q:.2f} ms |\n"
        f"| Schema DDL bootstrap | {ddl:.1f} ms |\n"
        f"| Seed insert | {seed/1000:.0f}K rows/sec |\n"
        f"| OLTP, 1 client | {tps['c1']/1000:.1f}K tps, p99 {p99['c1']:.0f} us |\n"
        f"| OLTP, 4 clients | {tps['c4']/1000:.1f}K tps, p99 {p99['c4']:.0f} us |\n"
        f"| OLTP, 16 clients | {tps['c16']/1000:.1f}K tps, p99 {p99['c16']:.0f} us |\n"
        f"| Analytical query (median) | {analytics_us/1000:.2f} ms |\n"
        f"| Graceful shutdown | {teardown:.0f} ms |\n"
    )
    tps_labels = [lbl for _, lbl in conns]
    tps_vals = [tps[c] / 1000 for c, _ in conns]
    tps_top = int(max(tps_vals) * 1.15)

    any_doc = next(iter(cache.values()))
    hw = f'{any_doc["cpu"]}, {any_doc["cores"]} cores, {any_doc["ram_gb"]} GB RAM, {any_doc["os"]}/{any_doc["arch"]}'

    tp_top = int(max(tp_vals) * 1.15)
    lat_top = int(max(lat_vals) * 1.15)

    out = []
    out.append(
        f"_Release build, single machine: {hw}. Regenerated from `benchmarks/` by `scripts/gen_bench_charts.py`._\n"
    )
    out.append("### End-to-end\n")
    out.append("What a client sees from a running server over the wire protocol, from cold start to shutdown:\n")
    out.append(e2e_table)
    out.append(bar_block("OLTP throughput vs. concurrent clients (thousand tps, higher is better)", "K tps", tps_labels, tps_vals, tps_top))
    out.append("### Engine internals\n")
    out.append("Raw subsystem throughput and hot-path latency under microbenchmark:\n")
    out.append(bar_block("Throughput (million ops/sec, higher is better)", "M ops/sec", tp_labels, tp_vals, tp_top))
    out.append(bar_block("Hot-path latency (nanoseconds, lower is better)", "ns/op", lat_labels, lat_vals, lat_top))

    extra_rows = []
    for suite, group, key, sub, mlabel, div, dec, suffix in EXTRA:
        v = metric(doc(suite), group, key) / div
        extra_rows.append(f"| {sub} | {mlabel} | ~{v:.{dec}f}{suffix} |")
    extra_table = (
        "A few more numbers not shown in the charts above:\n\n"
        "| Subsystem | Metric | Result |\n"
        "|-----------|--------|--------|\n" + "\n".join(extra_rows) + "\n"
    )
    out.append(extra_table)

    suite_count = sum(1 for p in BENCH.iterdir() if p.is_dir())
    out.append(
        f"{suite_count} benchmark suites cover storage, executor, optimizer, encoding, wire, "
        "search, analytics, CDC, versioning, transactions, types, lifecycle, and end-to-end. "
        "Each run writes a timestamped JSON/TXT pair under `benchmarks/<suite>/`.\n"
    )

    block = "<!-- BENCH:START -->\n" + "\n".join(out) + "<!-- BENCH:END -->"
    text = README.read_text(encoding="utf-8")
    new = re.sub(r"<!-- BENCH:START -->.*<!-- BENCH:END -->", block, text, flags=re.S)
    README.write_text(new, encoding="utf-8")
    print("README.md benchmark section regenerated.")


if __name__ == "__main__":
    main()
