//! The bottom of the crate graph stays at the bottom.
//!
//! `zyron-common` is depended on by everything. The pressure substrate used to
//! live inside it, which meant a control loop with a background tick and a
//! hardware probe sat underneath the type definitions, and the error type
//! could reach the controller. Both crates that came out of it sit above this
//! one now, and the way that stays true is a test, because the way it stops
//! being true is one convenient `use` that compiles.
//!
//! Checked three ways. The manifest, because that is what would have to change
//! for the dependency to exist. The source, because a path naming either crate
//! would not compile without the manifest but says plainly what someone was
//! reaching for. And the resolved graph, because a path dependency two crates
//! away can pull one back in without this manifest naming it.
//!
//! Run: cargo test -p zyron-common --test layering_guard_test

use std::path::{Path, PathBuf};

/// Crates that must never appear beneath this one.
const ABOVE: [&str; 2] = ["zyron-pressure", "zyron-mesh"];

fn manifest() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml")
}

fn source_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn sources(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }
    out
}

/// Nothing above this crate is a dependency of it.
///
/// The whole manifest is read rather than only the `[dependencies]` section,
/// because a dev-dependency would be enough to make `cargo test -p
/// zyron-common` build the thing this crate is supposed to sit under, and a
/// build-dependency would put it in the build graph.
#[test]
fn neither_crate_above_appears_in_this_manifest() {
    let text = std::fs::read_to_string(manifest()).expect("zyron-common manifest");
    for name in ABOVE {
        assert!(
            !text.contains(name),
            "{name} appears in zyron-common's manifest, which puts a crate that depends on \
             zyron-common underneath it"
        );
    }
}

/// And no source file names one.
///
/// A path here could not link without the manifest entry above, so this is
/// the earlier signal: it catches the reach before the manifest is edited to
/// allow it. Comments count, because a commented-out `use` is a plan.
#[test]
fn no_source_file_reaches_upward() {
    let mut offenders = Vec::new();
    for file in sources(&source_dir()) {
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        for (number, line) in text.lines().enumerate() {
            for name in ABOVE {
                let path_form = name.replace('-', "_");
                if line.contains(&path_form) {
                    offenders.push(format!(
                        "{}:{} {}",
                        file.file_name().unwrap_or_default().to_string_lossy(),
                        number + 1,
                        line.trim()
                    ));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "zyron-common reached up into a crate that sits above it:\n  {}",
        offenders.join("\n  ")
    );
}

/// And the resolved dependency graph agrees with the manifest.
///
/// The manifest is what someone edits, and the graph is what cargo builds.
/// They can disagree: a path dependency two crates away can pull one of these
/// back in without this manifest naming it. Normal edges only, because this
/// crate dev-depends on the benchmark harness, which sits near the top of the
/// tree and legitimately reaches both. A dev edge does not put anything under
/// this crate for anyone who depends on it, which is the property being
/// guarded.
#[test]
fn the_resolved_graph_puts_nothing_above_underneath() {
    let output = std::process::Command::new(env!("CARGO"))
        .args(["tree", "-p", "zyron-common", "--edges", "normal"])
        .current_dir(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(Path::parent)
                .expect("repo root"),
        )
        .output()
        .expect("cargo tree runs");
    assert!(
        output.status.success(),
        "cargo tree failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let tree = String::from_utf8_lossy(&output.stdout);
    for name in ABOVE {
        let hits: Vec<&str> = tree.lines().filter(|line| line.contains(name)).collect();
        assert!(
            hits.is_empty(),
            "{name} resolves underneath zyron-common:\n  {}",
            hits.join("\n  ")
        );
    }
}

/// The seam the substrate uses instead is still here.
///
/// Without this the two tests above would keep passing after somebody removed
/// the conflict signal entirely, which would be the layering intact and the
/// signal gone.
#[test]
fn the_conflict_seam_is_still_the_way_up() {
    let error = std::fs::read_to_string(source_dir().join("error.rs")).expect("error source");
    assert!(
        error.contains("conflict_signal::record_conflict_abort()"),
        "the transaction conflict constructor stopped reporting through the seam"
    );
    assert!(
        !zyron_common::conflict_signal::has_conflict_sink(),
        "something installed a conflict sink into a test that depends on nothing above this crate"
    );
}
