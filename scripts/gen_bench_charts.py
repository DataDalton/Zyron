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
CHARTS = BENCH / "charts"
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
    ("columnar", "Bloom Filter", "Bloom probe latency (ns)", "Bloom probe"),
    ("optimizer", "optimizer", "selectivity_estimate_ns", "Selectivity est"),
]

# Supplementary table, rendered below the charts.
# (suite, group, metric key, subsystem, metric label, divisor, decimals, suffix)
EXTRA = [
    (
        "transaction",
        "Phase 1.5 Microbenchmarks",
        "GC sweep throughput (tuples/sec)",
        "MVCC",
        "GC sweep",
        1e9,
        1,
        "B tuples/sec",
    ),
    (
        "columnar",
        "Column Segment Format",
        ".zyr scan throughput (GB/sec)",
        "Columnar",
        ".zyr scan throughput",
        1,
        1,
        " GB/sec",
    ),
    (
        "columnar",
        "Compaction Pipeline",
        "Compaction throughput (rows/sec)",
        "Columnar",
        "Compaction pipeline",
        1e6,
        1,
        "M rows/sec",
    ),
    (
        "columnar",
        "HTAP Hybrid Scan",
        "Hybrid scan overhead (%)",
        "Columnar",
        "HybridScan overhead vs heap-only",
        1,
        1,
        "%",
    ),
    (
        "columnar",
        "Metadata Aggregate",
        "Metadata vs scan speedup (x)",
        "Columnar",
        "Metadata-aggregate pruning speedup",
        1,
        1,
        "x",
    ),
    (
        "temporal",
        "ps_decode_throughput",
        "ps decode rows/sec",
        "Temporal",
        "Picosecond timestamp decode",
        1e6,
        0,
        "M rows/sec",
    ),
    (
        "versioning",
        "time_travel_scan_overhead",
        "overhead_percent",
        "Versioning",
        "Time-travel scan overhead",
        1,
        0,
        "%",
    ),
    ("wire", "QUIC PG Handshake", "QUIC PG handshake latency (us)", "Wire", "QUIC PostgreSQL handshake", 1, 0, " us"),
    (
        "transaction",
        "Phase 1.5 Microbenchmarks",
        "durable begin()+commit() latency floor (ns/op)",
        "Transactions",
        "Durable commit floor (device write)",
        1e3,
        1,
        " us",
    ),
    ("lake", "lake_scan", "Scan throughput", "Lake", "Scan throughput", 1e6, 0, "M rows/sec"),
    ("lake", "lake_index", "Point probe through the index", "Lake", "Point probe through index", 1, 0, " us"),
    ("lake", "lake_skipping", "zone map rows rejected (fraction)", "Lake", "Zone-map row rejection", 0.01, 0, "%"),
]

# Cross-format workloads to chart (heap vs lake wall-clock ratios).
# (group, key, short label) - values pulled from tests[group][key]["ratio"]["value"].
CROSS_READS = [
    ("point_lookup_narrow", "Point lookup, narrow projection", "Point lookup"),
    ("bulk_delete", "Bulk delete", "Bulk delete"),
    ("range_scan", "Range scan, selective", "Selective scan"),
    ("aggregate", "Aggregate over one column", "Aggregate"),
    ("point_update", "Point update", "Point update"),
    ("join", "Join", "Join"),
    ("range_scan", "Range scan, wide", "Wide scan"),
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


def _svg_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_grouped_bar_svg(path, title, x_labels, series, y_max, y_unit, value_fmt="{:,.0f}"):
    """Write a self-contained dark-themed SVG grouped bar chart. Mermaid's
    xychart-beta can't render side-by-side grouped bars, so this bypasses it.
    Each x-label gets one bar per series, colored per series.

    `series` is a list of (name, color_hex, values); all values lists must be
    the same length as x_labels.
    """
    n_groups = len(x_labels)
    n_series = len(series)
    # Layout constants (in SVG user units).
    W, H = 960, 460
    m_left, m_right, m_top, m_bot = 70, 30, 60, 110
    plot_w = W - m_left - m_right
    plot_h = H - m_top - m_bot
    group_w = plot_w / n_groups
    bar_w = min(48, (group_w - 30) / n_series)
    bar_gap = 4  # gap between bars within a group
    inner_w = bar_w * n_series + bar_gap * (n_series - 1)
    # Colors (self-contained dark theme so it reads on both GitHub themes).
    bg = "#0d1117"
    grid = "#30363d"
    axis = "#8b949e"
    text = "#e6edf3"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" font-family="ui-sans-serif, system-ui, sans-serif">',
        f'  <rect width="{W}" height="{H}" fill="{bg}" rx="6"/>',
        f'  <text x="{W/2}" y="30" text-anchor="middle" fill="{text}" font-size="16" font-weight="600">{_svg_escape(title)}</text>',
    ]

    # Y-axis: 5 tick marks 0..y_max.
    for i in range(6):
        y_val = y_max * i / 5
        y_px = m_top + plot_h - (plot_h * i / 5)
        lines.append(f'  <line x1="{m_left}" y1="{y_px:.1f}" x2="{W-m_right}" y2="{y_px:.1f}" stroke="{grid}" stroke-width="1"/>')
        lines.append(f'  <text x="{m_left-8}" y="{y_px+4:.1f}" text-anchor="end" fill="{axis}" font-size="11">{y_val:,.0f}</text>')
    # Y-axis unit label.
    lines.append(f'  <text x="{m_left-8}" y="{m_top-8}" text-anchor="end" fill="{axis}" font-size="11">{_svg_escape(y_unit)}</text>')

    # Bars + group labels.
    for gi, label in enumerate(x_labels):
        group_x = m_left + gi * group_w
        group_center = group_x + group_w / 2
        inner_start = group_center - inner_w / 2
        for si, (_, color, vals) in enumerate(series):
            v = vals[gi]
            bar_h = plot_h * (v / y_max) if y_max else 0
            bx = inner_start + si * (bar_w + bar_gap)
            by = m_top + plot_h - bar_h
            lines.append(f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="2"/>')
            # Value label above the bar.
            lines.append(f'  <text x="{bx + bar_w/2:.1f}" y="{by - 4:.1f}" text-anchor="middle" fill="{text}" font-size="10">{value_fmt.format(v)}</text>')
        # Group label below x-axis.
        lines.append(f'  <text x="{group_center:.1f}" y="{H - m_bot + 20}" text-anchor="middle" fill="{text}" font-size="12">{_svg_escape(label)}</text>')

    # Legend row at the bottom (only if there's more than one series to distinguish).
    if n_series > 1:
        legend_y = H - 40
        sw_size = 14
        entries = [(name, color) for name, color, _ in series]
        total_legend_w = sum(sw_size + 6 + 8 * len(name) + 24 for name, _ in entries) - 24
        lx = (W - total_legend_w) / 2
        for name, color in entries:
            lines.append(f'  <rect x="{lx:.1f}" y="{legend_y - sw_size + 2:.1f}" width="{sw_size}" height="{sw_size}" fill="{color}" rx="3"/>')
            lines.append(f'  <text x="{lx + sw_size + 6:.1f}" y="{legend_y:.1f}" fill="{text}" font-size="13">{_svg_escape(name)}</text>')
            lx += sw_size + 6 + 8 * len(name) + 24

    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


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

    # Durable group-commit: peak storage-engine commit throughput and the
    # serial-to-peak amplification factor. Levels are discovered from the
    # JSON so a change to CONCURRENCY_LEVELS flows through automatically.
    txn = doc("transaction")["tests"].get("Concurrent Txns", {})
    dc_levels = sorted(
        (int(m.group(1)) for k in txn if (m := re.fullmatch(r"durable_commit_c(\d+)_txn_per_sec", k))),
        key=int,
    )
    dc_peak = max(metric(doc("transaction"), "Concurrent Txns", f"durable_commit_c{n}_txn_per_sec") for n in dc_levels)
    dc_serial = metric(doc("transaction"), "Concurrent Txns", f"durable_commit_c{dc_levels[0]}_txn_per_sec")
    dc_amplification = dc_peak / dc_serial

    # Cross-format: heap vs lake ratios per workload. Read-shape ratios go in
    # the chart; the write losses and bytes-read wins go in the supplementary
    # table so the tradeoff is stated honestly rather than hidden.
    cf = doc("cross_format")

    def cf_ratio(group, key):
        return cf["tests"][group][key]["ratio"]["value"]

    def cf_us(group, key, fmt):
        # fmt is "heap" or "lake"; both are wall-clock microseconds.
        return cf["tests"][group][key][fmt]["average"]

    cf_labels = [label for _, _, label in CROSS_READS]
    cf_heap_us = [cf_us(g, k, "heap") for g, k, _ in CROSS_READS]
    cf_lake_us = [cf_us(g, k, "lake") for g, k, _ in CROSS_READS]
    cf_top = int(max(cf_heap_us + cf_lake_us) * 1.15)
    cf_bytes_reduction = 1.0 / cf_ratio("point_lookup_narrow", "Bytes read")
    cf_bulk_load = cf_ratio("bulk_load", "Bulk load to queryable")
    cf_trickle_load = cf_ratio("trickle_load", "Trickle load to queryable")
    cf_point_indexed = cf_ratio("point_lookup_index", "Point lookup with an index")

    # End-to-end: a client talking to a running server over the wire.
    e2e = doc("end_to_end")
    cold = metric(e2e, "Startup", "cold boot latency (ms)")
    first_q = metric(e2e, "Bootstrap", "first ReadyForQuery (ms)")
    ddl = metric(e2e, "Bootstrap", "schema DDL total (ms)")
    seed = metric(e2e, "Bootstrap", "seed insert (rows/sec)")
    teardown = metric(e2e, "Shutdown", "teardown (ms)")
    analytics_us = metric(e2e, "Analytics", "median query us")
    # Discover OLTP concurrency levels from the JSON itself so the bench can
    # change its OLTP_CONCURRENCY_LEVELS without the script needing edits.
    oltp = e2e["tests"].get("OLTP", {})
    levels = sorted(
        (int(m.group(1)) for k in oltp if (m := re.fullmatch(r"c(\d+) tps", k))),
        key=int,
    )
    tps = {n: metric(e2e, "OLTP", f"c{n} tps") for n in levels}
    p99 = {n: metric(e2e, "OLTP", f"c{n} p99 us") for n in levels}

    oltp_rows = "".join(
        f"| OLTP, {n} client{'s' if n != 1 else ''} | {tps[n]/1000:.1f}K tps, p99 {p99[n]:.0f} us |\n" for n in levels
    )
    e2e_table = (
        "| Lifecycle / workload | Result |\n"
        "|----------------------|--------|\n"
        f"| Cold boot to accepting queries | {cold:.0f} ms |\n"
        f"| First `ReadyForQuery` | {first_q:.2f} ms |\n"
        f"| Schema DDL bootstrap | {ddl:.1f} ms |\n"
        f"| Seed insert | {seed/1000:.0f}K rows/sec |\n"
        f"{oltp_rows}"
        f"| Analytical query (median) | {analytics_us/1000:.2f} ms |\n"
        f"| Graceful shutdown | {teardown:.0f} ms |\n"
    )
    tps_labels = [f"{n} client{'s' if n != 1 else ''}" for n in levels]
    tps_vals = [tps[n] / 1000 for n in levels]
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
    def svg_chart(name, title, labels, values, top, unit, color="#1f6feb", fmt="{:.1f}"):
        path = CHARTS / f"{name}.svg"
        write_grouped_bar_svg(path, title, labels, [("", color, values)], top, unit, value_fmt=fmt)
        rel = path.relative_to(ROOT).as_posix()
        out.append(f"![{title}]({rel})\n")

    svg_chart(
        "oltp_throughput",
        "OLTP throughput vs. concurrent clients (thousand tps, higher is better)",
        tps_labels,
        tps_vals,
        tps_top,
        "K tps",
    )
    out.append("### Engine internals\n")
    out.append("Raw subsystem throughput and hot-path latency under microbenchmark:\n")
    svg_chart(
        "engine_throughput",
        "Throughput (million ops/sec, higher is better)",
        tp_labels,
        tp_vals,
        tp_top,
        "M ops/sec",
    )
    svg_chart(
        "hot_path_latency",
        "Hot-path latency (nanoseconds, lower is better)",
        lat_labels,
        lat_vals,
        lat_top,
        "ns/op",
    )
    out.append("### Row heap vs ZyronLake\n")
    out.append(
        "Same workload, same rows, run against both formats. Blue = Row heap, purple = "
        "ZyronLake; the shorter bar wins on wall-clock. Write-heavy trade-offs where the "
        "Row heap wins (bulk load, trickle load, indexed point lookup) are in the table "
        "below so the picture is honest, not cherry-picked.\n"
    )
    svg_top = int(max(cf_heap_us + cf_lake_us) * 1.15)
    svg_path = CHARTS / "cross_format.svg"
    write_grouped_bar_svg(
        svg_path,
        "Cross-format wall-clock (microseconds, lower is better)",
        cf_labels,
        [
            ("Row heap", "#1f6feb", cf_heap_us),
            ("ZyronLake", "#a371f7", cf_lake_us),
        ],
        svg_top,
        "us",
    )
    svg_rel = svg_path.relative_to(ROOT).as_posix()
    out.append(f"![Cross-format wall-clock, Row heap vs ZyronLake]({svg_rel})\n")

    extra_rows = []
    for suite, group, key, sub, mlabel, div, dec, suffix in EXTRA:
        v = metric(doc(suite), group, key) / div
        extra_rows.append(f"| {sub} | {mlabel} | ~{v:.{dec}f}{suffix} |")
    extra_rows.append(f"| Transactions | Durable group-commit peak | ~{dc_peak/1000:.0f}K txn/sec |")
    extra_rows.append(
        f"| Transactions | Group-commit amplification (c={dc_levels[0]} to c={dc_levels[-1]}) | ~{dc_amplification:.1f}x |"
    )
    extra_rows.append(f"| Cross-format | Point-lookup bytes read, heap vs lake | ~{cf_bytes_reduction:.0f}x less I/O for lake |")
    extra_rows.append(f"| Cross-format | Point lookup with a heap B+tree index, lake vs heap | ~{cf_point_indexed:.1f}x (heap wins indexed points) |")
    extra_rows.append(f"| Cross-format | Bulk load to queryable, lake vs heap | ~{cf_bulk_load:.1f}x (heap wins large batches) |")
    extra_rows.append(f"| Cross-format | Trickle load to queryable, lake vs heap | ~{cf_trickle_load:.1f}x (heap wins tiny commits) |")
    extra_table = (
        "A few more numbers not shown in the charts above:\n\n"
        "| Subsystem | Metric | Result |\n"
        "|-----------|--------|--------|\n" + "\n".join(extra_rows) + "\n"
    )
    out.append(extra_table)

    suite_count = sum(1 for p in BENCH.iterdir() if p.is_dir())
    out.append(
        f"{suite_count} benchmark suites cover storage, executor, optimizer, encoding, wire, "
        "search, analytics, CDC, versioning, transactions, temporal, columnar, lake, "
        "cross-format, types, lifecycle, gateway, Zyron-to-Zyron, and end-to-end. Each run "
        "writes a timestamped "
        "JSON/TXT pair under `benchmarks/<suite>/`.\n"
    )

    block = "<!-- BENCH:START -->\n" + "\n".join(out) + "<!-- BENCH:END -->"
    text = README.read_text(encoding="utf-8")
    new = re.sub(r"<!-- BENCH:START -->.*<!-- BENCH:END -->", block, text, flags=re.S)
    README.write_text(new, encoding="utf-8")
    print("README.md benchmark section regenerated.")


if __name__ == "__main__":
    main()
