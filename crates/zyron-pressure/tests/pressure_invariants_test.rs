//! Guards against the three mistakes this substrate exists to remove.
//!
//! Each one was true of the tree before, each is easy to reintroduce by
//! reaching for the obvious thing, and none of them fails a normal test:
//! spawning onto the wrong runtime still returns the right answer, sizing
//! concurrency from the core count still serves queries, and reading the
//! provisioner mode in the scaling logic still scales. They fail only under
//! load, or on a machine that is not this one, which is exactly when nobody
//! is watching. So they are checked against the source.
//!
//! Run: cargo test -p zyron-pressure --test pressure_invariants_test

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is crates/zyron-pressure
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("repo root")
        .to_path_buf()
}

/// Every .rs file under a path, excluding build output and editor history.
fn sources(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if path.is_dir() {
                if name == "target" || name == ".history" || name == ".git" {
                    continue;
                }
                stack.push(path);
            } else if name.ends_with(".rs") {
                out.push(path);
            }
        }
    }
    out
}

/// Lines of a file that contain `needle`, outside any `#[cfg(test)]` module
/// and outside comments.
///
/// The test module is excluded because a test proving the rule necessarily
/// names the thing the rule forbids, and a comment is excluded because
/// explaining why something is forbidden is not doing it.
fn offending_lines(path: &Path, needle: &str) -> Vec<(usize, String)> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut in_tests = false;
    let mut test_depth = 0i32;
    for (i, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("#[cfg(test)]") {
            in_tests = true;
            test_depth = 0;
            continue;
        }
        if in_tests {
            test_depth += line.matches('{').count() as i32;
            test_depth -= line.matches('}').count() as i32;
            if test_depth <= 0 && line.contains('}') {
                in_tests = false;
            }
            continue;
        }
        if trimmed.starts_with("//") {
            continue;
        }
        if line.contains(needle) {
            out.push((i + 1, line.trim().to_string()));
        }
    }
    out
}

fn rel(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// The serving path must not size parallel work from the core count.
///
/// It is the right number for exactly one thing, the size of the parallel
/// work pool, and the wrong number everywhere else: a scan that asks for
/// `cores` workers is asking for the whole machine regardless of how many
/// other queries are already running.
#[test]
fn no_available_parallelism_on_the_serving_path() {
    let root = repo_root();
    let mut offenders = Vec::new();
    // Not this crate's own source: the probe is where the machine is read
    // once, and the parallel budget is sized from what it found. The rule is
    // about the path that serves a query, and the test below covers the
    // controller specifically
    for crate_dir in ["crates/zyron-executor/src", "crates/zyron-wire/src"] {
        for file in sources(&root.join(crate_dir)) {
            // The pool constructor is where the machine is allowed to be read.
            // That is the probe, and the whole point is that it happens once
            if rel(&root, &file).ends_with("zyron-executor/src/parallel_pool.rs") {
                continue;
            }
            for (line, text) in offending_lines(&file, "available_parallelism") {
                offenders.push(format!("{}:{} {}", rel(&root, &file), line, text));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "the serving path sized itself from the core count again:\n  {}",
        offenders.join("\n  ")
    );
}

/// The controller must never bound query concurrency by the core count.
///
/// A query parked on storage has released its thread, so one thread hosts
/// many in-flight queries and a ceiling of `cores` would idle the machine
/// waiting on disk. The ceiling is found by measurement instead.
#[test]
fn the_controller_never_reads_the_core_count() {
    let root = repo_root();
    let controller = root.join("crates/zyron-pressure/src/pressure_control.rs");
    let mut offenders = Vec::new();
    for needle in [
        "available_parallelism",
        "num_cpus",
        "hardware_concurrency",
        "core_count",
    ] {
        for (line, text) in offending_lines(&controller, needle) {
            offenders.push(format!("pressure_control.rs:{line} {text}"));
        }
    }
    assert!(
        offenders.is_empty(),
        "the controller went back to bounding concurrency by cores:\n  {}",
        offenders.join("\n  ")
    );
}

/// Whether a line that names the registration mode steers on it, rather than
/// only storing, returning or printing it.
///
/// A line whose trimmed form opens with a quote is a match arm or a list
/// entry keyed by the setting's name, which selects a setting rather than a
/// provisioner. Everything else counts as steering as soon as it compares
/// the value or opens a branch on it.
fn acts_on_the_value(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.starts_with('"') {
        return false;
    }
    const STEERING: [&str; 7] = [
        "==",
        "!=",
        "if ",
        "match ",
        ".eq(",
        ".starts_with(",
        ".contains(",
    ];
    STEERING.iter().any(|token| trimmed.contains(token))
}

/// How a node comes into existence must not decide how the node scales.
///
/// Every mode autoscales. Only the actuator and the ceiling differ, and both
/// belong to the provisioner. A branch anywhere else that reads the
/// registration mode is a static deployment quietly losing its elasticity.
///
/// The rule has two halves, because the mode has to exist somewhere. Outside
/// the provisioner nothing may act on the value, which is this test. Inside
/// it, only the places that declare, default, validate, and resolve it may
/// name it, which is the test below.
///
/// Acting on the value means comparing it or steering on it. Carrying it is
/// something else: an operator sets `mesh.node_registration_mode` and reads
/// it back, so the config surface stores the string, hands it out and lists
/// it, and a startup log names the mode the node came up in. None of those
/// decide anything, and forbidding them would mean the setting could not be
/// set. What they must never become is a second answer to which provisioner
/// runs, because the registry already owns that question.
#[test]
fn no_registration_mode_branch_outside_the_provisioner() {
    let root = repo_root();
    let mut offenders = Vec::new();
    for crate_dir in [
        "crates/zyron-common/src",
        "crates/zyron-pressure/src",
        "crates/zyron-mesh/src",
        "crates/zyron-executor/src",
        "crates/zyron-planner/src",
        "crates/zyron-buffer/src",
        "crates/zyron-wire/src",
        "crates/zyron-server/src",
        "binaries",
    ] {
        for file in sources(&root.join(crate_dir)) {
            let name = rel(&root, &file);
            // The provisioner is the one place allowed to ask
            if name.ends_with("zyron-pressure/src/provisioner.rs") {
                continue;
            }
            for (line, text) in offending_lines(&file, "node_registration_mode") {
                if !acts_on_the_value(&text) {
                    continue;
                }
                offenders.push(format!("{name}:{line} {text}"));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "the scaling path branched on how the node was provisioned:\n  {}",
        offenders.join("\n  ")
    );
}

/// Inside the provisioner, the mode reaches exactly the places that resolve it.
///
/// Declaring the field, defaulting it, refusing one nobody implements, and
/// turning it into a driver. A fifth reader is the beginning of a special
/// case, and the reason this module exists is that there are none: everything
/// else sees a driver with capabilities.
#[test]
fn inside_the_provisioner_the_mode_only_selects_a_driver() {
    let root = repo_root();
    let provisioner = root.join("crates/zyron-pressure/src/provisioner.rs");
    let text = std::fs::read_to_string(&provisioner).expect("provisioner source");

    let allowed = [
        "pub node_registration_mode: String,",
        "node_registration_mode: \"none\".into(),",
        "if ProvisionerKind::parse(&self.node_registration_mode).is_none() {",
        "name: \"mesh.node_registration_mode\".into(),",
        "ProvisionerRegistry::global().select(&self.node_registration_mode)",
        "pub fn select(&self, node_registration_mode: &str) -> Arc<dyn ProvisionerDriver> {",
        "let kind = ProvisionerKind::parse(node_registration_mode).unwrap_or(ProvisionerKind::None);",
        // Naming the key back to the operator is reporting, not branching
        "value: self.node_registration_mode.clone(),",
    ];

    let mut unexpected = Vec::new();
    let mut in_tests = false;
    let mut depth = 0i32;
    for (i, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("#[cfg(test)]") {
            in_tests = true;
            depth = 0;
            continue;
        }
        if in_tests {
            depth += line.matches('{').count() as i32;
            depth -= line.matches('}').count() as i32;
            if depth <= 0 && line.contains('}') {
                in_tests = false;
            }
            continue;
        }
        if trimmed.starts_with("//") {
            continue;
        }
        // A message that names the key is telling an operator which setting
        // to look at, which is the opposite of a hidden special case
        if trimmed.starts_with('"') {
            continue;
        }
        if !line.contains("node_registration_mode") {
            continue;
        }
        if !allowed.contains(&trimmed) {
            unexpected.push(format!("{}: {trimmed}", i + 1));
        }
    }

    assert!(
        unexpected.is_empty(),
        "the registration mode reached somewhere new inside the provisioner:\n  {}",
        unexpected.join("\n  ")
    );
}

/// The threshold-with-cooldown scaler it replaced must stay gone.
///
/// Running both would give the node two opinions about whether it is busy,
/// and the quieter one would win by acting first.
#[test]
fn the_old_threshold_scaler_is_gone() {
    let root = repo_root();
    let mut offenders = Vec::new();
    for crate_dir in [
        "crates/zyron-common/src",
        "crates/zyron-pressure/src",
        "crates/zyron-mesh/src",
        "crates/zyron-streaming/src",
        "crates/zyron-executor/src",
        "crates/zyron-wire/src",
        "crates/zyron-server/src",
    ] {
        for file in sources(&root.join(crate_dir)) {
            for needle in ["AutoScaling", "ScalingDecision"] {
                for (line, text) in offending_lines(&file, needle) {
                    offenders.push(format!("{}:{} {}", rel(&root, &file), line, text));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "the replaced scaler is still referenced:\n  {}",
        offenders.join("\n  ")
    );
}

/// A fixed connection count cannot be reintroduced through the config.
///
/// The ceiling is what the node's memory affords. A configured number is a
/// limit the hardware never asked for, and the failure being guarded against
/// is a hundred-connection default nobody could raise past their own machine.
#[test]
fn the_connection_count_is_not_configurable() {
    let root = repo_root();
    let common_config = root.join("crates/zyron-common/src/config.rs");
    let offenders = offending_lines(&common_config, "max_connections");
    assert!(
        offenders.is_empty(),
        "a connection count came back to ServerConfig:\n  {:?}",
        offenders
    );
}

/// The gossip payload must keep its sequence number.
///
/// The transport converges by taking a maximum, which only converges on a
/// value that never decreases. Pressure falls all the time, so without the
/// sequence in the high half a node that recovered would keep advertising its
/// worst reading forever.
#[test]
fn the_gossip_encoding_is_monotone() {
    use zyron_pressure::pressure::{decode_versioned, encode_versioned};
    let high_then_low = [
        encode_versioned(1, u32::MAX),
        encode_versioned(2, 0),
        encode_versioned(3, 5),
    ];
    for pair in high_then_low.windows(2) {
        assert!(
            pair[1] > pair[0],
            "a later tick compared lower, so a max merge would keep the stale one"
        );
    }
    assert_eq!(decode_versioned(high_then_low[1]), (2, 0));
}
