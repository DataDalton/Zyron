#![allow(non_snake_case)]
//! Zyron cluster management and admin tool.
//!
//! Provides subcommands for server status, backup, restore, checkpoint,
//! analyze, vacuum, compaction, configuration validation, and benchmarks.

mod bench;
mod docs_cmd;
mod format_cmd;
mod function_index;
mod remote;

use std::path::{Path, PathBuf};
use std::process;

use remote::RemoteClient;
use zyron_server::backup::{BackupManager, RestoreManager};
use zyron_server::config::ZyronConfig;

const VERSION: &str = env!("CARGO_PKG_VERSION");

// ---------------------------------------------------------------------------
// CLI data types
// ---------------------------------------------------------------------------

struct GlobalFlags {
    host: String,
    port: u16,
    user: String,
    database: String,
}

impl Default for GlobalFlags {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 5432,
            user: "zyron".into(),
            database: "zyron".into(),
        }
    }
}

enum Subcommand {
    Status,
    Backup {
        dataDir: PathBuf,
        output: PathBuf,
    },
    Restore {
        input: PathBuf,
        dataDir: PathBuf,
    },
    Checkpoint,
    Analyze {
        table: Option<String>,
    },
    Vacuum {
        table: Option<String>,
    },
    Compact {
        table: Option<String>,
    },
    ConfigValidate {
        config: PathBuf,
    },
    BenchTpch {
        scale: f64,
    },
    BenchTpcc {
        scale: f64,
    },

    // Format substrate. The three format commands read files directly, so
    // they work on a node that will not start
    FormatInspect {
        file: PathBuf,
    },
    FormatMigrate {
        format: String,
        to: Option<String>,
        path: PathBuf,
    },
    FormatVerify {
        path: PathBuf,
    },

    // Upgrade. These go to a running server, because upgrade state lives
    // there rather than on disk
    UpgradeCheck {
        verbose: bool,
    },
    UpgradeTrigger {
        version: String,
    },
    UpgradeRollback,
    UpgradePause,
    UpgradeResume,
    UpgradeShowState,
    UpgradeAcknowledge {
        category: String,
    },

    DocsGenerate {
        root: PathBuf,
    },
    DocsCheck {
        root: PathBuf,
    },
    DeprecationReport,
    DeprecationGuide {
        item: String,
    },

    ReleaseVerify {
        today: Option<String>,
    },
    /// Puts a signed manifest and a release binary where an air-gapped
    /// node's feed reads them
    ReleaseStage {
        manifest: PathBuf,
        binary: PathBuf,
        dataDir: PathBuf,
        version: Option<String>,
    },
    /// Draws a release signing key and prints its public half
    ReleaseKeygen {
        out: PathBuf,
    },
    /// Signs a binary into a release manifest
    ReleaseSign {
        key: PathBuf,
        binary: PathBuf,
        version: String,
        manifest: Option<PathBuf>,
        channel: Option<String>,
        artifactUrl: Option<String>,
        notesUrl: Option<String>,
        chain: Vec<String>,
        formatBump: bool,
        out: PathBuf,
    },
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

fn printHelp() {
    println!(
        "Zyron {} - cluster management tool

Usage: zyron-ctl [global-flags] <subcommand> [subcommand-flags]

Global flags (for commands that connect to a server):
  --host <host>       Server host (default: 127.0.0.1)
  --port <port>       Server port (default: 5432)
  --user <user>       Username (default: zyron)
  --database <db>     Database (default: zyron)

Subcommands:
  status                                    Show active sessions and server health
  backup --data-dir <path> --output <path>  Create physical backup (offline)
  restore --input <path> --data-dir <path>  Restore from backup (offline)
  checkpoint                                Force immediate checkpoint
  analyze [table]                           Run ANALYZE on table or all tables
  vacuum [table]                            Run VACUUM on table or all tables
  compact [table]                           Force compaction (OPTIMIZE TABLE)
  config validate --config <path>           Validate configuration file
  bench tpch --scale <N>                    Run TPC-H benchmark
  bench tpcc --scale <N>                    Run TPC-C benchmark

Format subcommands (read files directly, no server needed):
  format inspect <file>                     Report a file's format kind, version, integrity
  format migrate --format <kind> [--to <version>] --path <dir>
                                            Move every file of one kind forward
  format verify --path <dir>                Check every Zyron file against the registry

Upgrade subcommands (connect to a server):
  upgrade check [--verbose]                 Dry-run the compatibility gate
  upgrade trigger --version <X.Y.Z>         Trigger an upgrade
  upgrade rollback                          Roll back when eligible
  upgrade pause                             Halt in-progress and queued upgrades
  upgrade resume                            Let upgrades proceed again
  upgrade show-state                        Show the current upgrade phase per node
  upgrade acknowledge --category <ambiguous|unsafe>
                                            Acknowledge the rewrites an upgrade waits on
  deprecation report                        List deprecation warnings in the trailing window
  deprecation guides <item>                 Print one item's migration guide

Release subcommands:
  release verify [--today <YYYY-MM-DD>]     Check every format bump, fixture, and retirement
                                            in this binary
  release stage --manifest <file> --binary <file> --data-dir <dir> [--version <X.Y.Z>]
                                            Put a signed manifest and a release binary where an
                                            air-gapped node's feed reads them
  release keygen --out <file>               Draw a release signing key into a file and print
                                            its public half
  release sign --key <file> --binary <file> --version <X.Y.Z> --out <file>
               [--manifest <file>] [--channel <name>] [--artifact-url <url>]
               [--notes-url <url>] [--chain <a,b>] [--format-bump]
                                            Sign a binary into a release manifest, adding to
                                            an existing manifest or starting one

Flags:
  --help                                    Print this help
  --version                                 Print version",
        VERSION
    );
}

fn parseArgs() -> Option<(GlobalFlags, Subcommand)> {
    let args: Vec<String> = std::env::args().collect();
    let mut flags = GlobalFlags::default();
    let mut i = 1;

    // Parse global flags first
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                printHelp();
                return None;
            }
            "--version" | "-V" => {
                println!("Zyron {} (zyron-ctl)", VERSION);
                return None;
            }
            "--host" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--host requires a value");
                    process::exit(1);
                }
                flags.host = args[i].clone();
            }
            "--port" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--port requires a value");
                    process::exit(1);
                }
                flags.port = match args[i].parse() {
                    Ok(p) => p,
                    Err(_) => {
                        eprintln!("invalid port: {}", args[i]);
                        process::exit(1);
                    }
                };
            }
            "--user" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--user requires a value");
                    process::exit(1);
                }
                flags.user = args[i].clone();
            }
            "--database" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--database requires a value");
                    process::exit(1);
                }
                flags.database = args[i].clone();
            }
            _ => break, // Start of subcommand
        }
        i += 1;
    }

    if i >= args.len() {
        eprintln!("No subcommand specified. Use --help for usage.");
        process::exit(1);
    }

    let subcmd = match args[i].as_str() {
        "status" => Subcommand::Status,

        "backup" => {
            i += 1;
            let mut dataDir: Option<PathBuf> = None;
            let mut output: Option<PathBuf> = None;
            while i < args.len() {
                match args[i].as_str() {
                    "--data-dir" => {
                        i += 1;
                        if i >= args.len() {
                            eprintln!("--data-dir requires a value");
                            process::exit(1);
                        }
                        dataDir = Some(PathBuf::from(&args[i]));
                    }
                    "--output" => {
                        i += 1;
                        if i >= args.len() {
                            eprintln!("--output requires a value");
                            process::exit(1);
                        }
                        output = Some(PathBuf::from(&args[i]));
                    }
                    other => {
                        eprintln!("Unknown backup flag: {}", other);
                        process::exit(1);
                    }
                }
                i += 1;
            }
            let dataDir = match dataDir {
                Some(d) => d,
                None => {
                    eprintln!("backup requires --data-dir <path>");
                    process::exit(1);
                }
            };
            let output = match output {
                Some(o) => o,
                None => {
                    eprintln!("backup requires --output <path>");
                    process::exit(1);
                }
            };
            Subcommand::Backup { dataDir, output }
        }

        "restore" => {
            i += 1;
            let mut input: Option<PathBuf> = None;
            let mut dataDir: Option<PathBuf> = None;
            while i < args.len() {
                match args[i].as_str() {
                    "--input" => {
                        i += 1;
                        if i >= args.len() {
                            eprintln!("--input requires a value");
                            process::exit(1);
                        }
                        input = Some(PathBuf::from(&args[i]));
                    }
                    "--data-dir" => {
                        i += 1;
                        if i >= args.len() {
                            eprintln!("--data-dir requires a value");
                            process::exit(1);
                        }
                        dataDir = Some(PathBuf::from(&args[i]));
                    }
                    other => {
                        eprintln!("Unknown restore flag: {}", other);
                        process::exit(1);
                    }
                }
                i += 1;
            }
            let input = match input {
                Some(p) => p,
                None => {
                    eprintln!("restore requires --input <path>");
                    process::exit(1);
                }
            };
            let dataDir = match dataDir {
                Some(d) => d,
                None => {
                    eprintln!("restore requires --data-dir <path>");
                    process::exit(1);
                }
            };
            Subcommand::Restore { input, dataDir }
        }

        "checkpoint" => Subcommand::Checkpoint,

        "analyze" => {
            i += 1;
            let table = if i < args.len() && !args[i].starts_with('-') {
                Some(args[i].clone())
            } else {
                None
            };
            Subcommand::Analyze { table }
        }

        "vacuum" => {
            i += 1;
            let table = if i < args.len() && !args[i].starts_with('-') {
                Some(args[i].clone())
            } else {
                None
            };
            Subcommand::Vacuum { table }
        }

        "compact" => {
            i += 1;
            let table = if i < args.len() && !args[i].starts_with('-') {
                Some(args[i].clone())
            } else {
                None
            };
            Subcommand::Compact { table }
        }

        "config" => {
            i += 1;
            if i >= args.len() || args[i] != "validate" {
                eprintln!("Usage: zyron-ctl config validate --config <path>");
                process::exit(1);
            }
            i += 1;
            let mut configPath: Option<PathBuf> = None;
            while i < args.len() {
                match args[i].as_str() {
                    "--config" => {
                        i += 1;
                        if i >= args.len() {
                            eprintln!("--config requires a value");
                            process::exit(1);
                        }
                        configPath = Some(PathBuf::from(&args[i]));
                    }
                    other => {
                        eprintln!("Unknown config validate flag: {}", other);
                        process::exit(1);
                    }
                }
                i += 1;
            }
            let config = match configPath {
                Some(c) => c,
                None => {
                    eprintln!("config validate requires --config <path>");
                    process::exit(1);
                }
            };
            Subcommand::ConfigValidate { config }
        }

        "bench" => {
            i += 1;
            if i >= args.len() {
                eprintln!("Usage: zyron-ctl bench <tpch|tpcc> --scale <N>");
                process::exit(1);
            }
            let benchType = args[i].clone();
            i += 1;
            let mut scale: f64 = 1.0;
            while i < args.len() {
                match args[i].as_str() {
                    "--scale" => {
                        i += 1;
                        if i >= args.len() {
                            eprintln!("--scale requires a value");
                            process::exit(1);
                        }
                        scale = match args[i].parse() {
                            Ok(s) => s,
                            Err(_) => {
                                eprintln!("invalid scale factor: {}", args[i]);
                                process::exit(1);
                            }
                        };
                    }
                    other => {
                        eprintln!("Unknown bench flag: {}", other);
                        process::exit(1);
                    }
                }
                i += 1;
            }
            match benchType.as_str() {
                "tpch" => Subcommand::BenchTpch { scale },
                "tpcc" => Subcommand::BenchTpcc { scale },
                other => {
                    eprintln!("Unknown benchmark type: {}. Use tpch or tpcc.", other);
                    process::exit(1);
                }
            }
        }
        "format" => parseFormat(&args, &mut i),
        "upgrade" => parseUpgrade(&args, &mut i),
        "deprecation" => parseDeprecation(&args, &mut i),
        "docs" => parseDocs(&args, &mut i),
        "release" => parseRelease(&args, &mut i),

        other => {
            eprintln!("Unknown subcommand: {}. Use --help for usage.", other);
            process::exit(1);
        }
    };

    Some((flags, subcmd))
}

// ---------------------------------------------------------------------------
// Subcommand handlers
// ---------------------------------------------------------------------------

/// Connects to the server and runs a SQL statement, printing the result table.
// ---------------------------------------------------------------------------
// Format substrate argument parsing
// ---------------------------------------------------------------------------

/// Reads the value of a flag, exiting with what was expected when it is
/// missing. Shared by the four parsers below so a missing value reads the
/// same whichever command it was on.
fn flagValue(args: &[String], i: &mut usize, flag: &str) -> String {
    *i += 1;
    if *i >= args.len() {
        eprintln!("{flag} requires a value");
        process::exit(1);
    }
    args[*i].clone()
}

fn parseFormat(args: &[String], i: &mut usize) -> Subcommand {
    *i += 1;
    if *i >= args.len() {
        eprintln!("Usage: zyron-ctl format <inspect|migrate|verify> ...");
        process::exit(1);
    }
    let action = args[*i].clone();
    *i += 1;
    match action.as_str() {
        "inspect" => {
            if *i >= args.len() {
                eprintln!("Usage: zyron-ctl format inspect <file>");
                process::exit(1);
            }
            let file = PathBuf::from(&args[*i]);
            *i += 1;
            Subcommand::FormatInspect { file }
        }
        "migrate" => {
            let mut format = None;
            let mut to = None;
            let mut path = None;
            while *i < args.len() {
                match args[*i].as_str() {
                    "--format" => format = Some(flagValue(args, i, "--format")),
                    "--to" => to = Some(flagValue(args, i, "--to")),
                    "--path" => path = Some(PathBuf::from(flagValue(args, i, "--path"))),
                    other => {
                        eprintln!("Unknown format migrate flag: {other}");
                        process::exit(1);
                    }
                }
                *i += 1;
            }
            match (format, path) {
                (Some(format), Some(path)) => Subcommand::FormatMigrate { format, to, path },
                _ => {
                    eprintln!(
                        "Usage: zyron-ctl format migrate --format <kind> [--to <version>] \
                         --path <dir>"
                    );
                    process::exit(1);
                }
            }
        }
        "verify" => {
            let mut path = None;
            while *i < args.len() {
                match args[*i].as_str() {
                    "--path" => path = Some(PathBuf::from(flagValue(args, i, "--path"))),
                    other => {
                        eprintln!("Unknown format verify flag: {other}");
                        process::exit(1);
                    }
                }
                *i += 1;
            }
            match path {
                Some(path) => Subcommand::FormatVerify { path },
                None => {
                    eprintln!("Usage: zyron-ctl format verify --path <dir>");
                    process::exit(1);
                }
            }
        }
        other => {
            eprintln!("Unknown format subcommand: {other}. Use inspect, migrate, or verify.");
            process::exit(1);
        }
    }
}

fn parseUpgrade(args: &[String], i: &mut usize) -> Subcommand {
    *i += 1;
    if *i >= args.len() {
        eprintln!("Usage: zyron-ctl upgrade <check|trigger|rollback|pause|resume|show-state> ...");
        process::exit(1);
    }
    let action = args[*i].clone();
    *i += 1;
    match action.as_str() {
        "check" => {
            let mut verbose = false;
            while *i < args.len() {
                match args[*i].as_str() {
                    "--verbose" => verbose = true,
                    other => {
                        eprintln!("Unknown upgrade check flag: {other}");
                        process::exit(1);
                    }
                }
                *i += 1;
            }
            Subcommand::UpgradeCheck { verbose }
        }
        "trigger" => {
            let mut version = None;
            while *i < args.len() {
                match args[*i].as_str() {
                    "--version" => version = Some(flagValue(args, i, "--version")),
                    other => {
                        eprintln!("Unknown upgrade trigger flag: {other}");
                        process::exit(1);
                    }
                }
                *i += 1;
            }
            match version {
                Some(version) => Subcommand::UpgradeTrigger { version },
                None => {
                    eprintln!("Usage: zyron-ctl upgrade trigger --version <X.Y.Z>");
                    process::exit(1);
                }
            }
        }
        "rollback" => Subcommand::UpgradeRollback,
        "pause" => Subcommand::UpgradePause,
        "resume" => Subcommand::UpgradeResume,
        "show-state" => Subcommand::UpgradeShowState,
        "acknowledge" => {
            let mut category = None;
            while *i < args.len() {
                match args[*i].as_str() {
                    "--category" => category = Some(flagValue(args, i, "--category")),
                    other => {
                        eprintln!("Unknown upgrade acknowledge flag: {other}");
                        process::exit(1);
                    }
                }
                *i += 1;
            }
            match category {
                Some(category)
                    if category.eq_ignore_ascii_case("ambiguous")
                        || category.eq_ignore_ascii_case("unsafe") =>
                {
                    Subcommand::UpgradeAcknowledge {
                        category: category.to_ascii_lowercase(),
                    }
                }
                _ => {
                    eprintln!("Usage: zyron-ctl upgrade acknowledge --category <ambiguous|unsafe>");
                    process::exit(1);
                }
            }
        }
        other => {
            eprintln!(
                "Unknown upgrade subcommand: {other}. Use check, trigger, rollback, pause, \
                 resume, show-state, or acknowledge."
            );
            process::exit(1);
        }
    }
}

fn parseDeprecation(args: &[String], i: &mut usize) -> Subcommand {
    *i += 1;
    if *i >= args.len() {
        eprintln!("Usage: zyron-ctl deprecation <report|guides> ...");
        process::exit(1);
    }
    let action = args[*i].clone();
    *i += 1;
    match action.as_str() {
        "report" => Subcommand::DeprecationReport,
        "guides" => {
            if *i >= args.len() {
                eprintln!("Usage: zyron-ctl deprecation guides <item>");
                process::exit(1);
            }
            let item = args[*i].clone();
            *i += 1;
            Subcommand::DeprecationGuide { item }
        }
        other => {
            eprintln!("Unknown deprecation subcommand: {other}. Use report or guides.");
            process::exit(1);
        }
    }
}

fn parseRelease(args: &[String], i: &mut usize) -> Subcommand {
    *i += 1;
    if *i >= args.len() {
        eprintln!("Usage: zyron-ctl release <verify|stage> ...");
        process::exit(1);
    }
    let action = args[*i].clone();
    *i += 1;
    match action.as_str() {
        "verify" => {
            let mut today = None;
            while *i < args.len() {
                match args[*i].as_str() {
                    "--today" => today = Some(flagValue(args, i, "--today")),
                    other => {
                        eprintln!("Unknown release verify flag: {other}");
                        process::exit(1);
                    }
                }
                *i += 1;
            }
            Subcommand::ReleaseVerify { today }
        }
        "stage" => {
            let mut manifest = None;
            let mut binary = None;
            let mut dataDir = None;
            let mut version = None;
            while *i < args.len() {
                match args[*i].as_str() {
                    "--manifest" => {
                        manifest = Some(PathBuf::from(flagValue(args, i, "--manifest")))
                    }
                    "--binary" => binary = Some(PathBuf::from(flagValue(args, i, "--binary"))),
                    "--data-dir" => dataDir = Some(PathBuf::from(flagValue(args, i, "--data-dir"))),
                    "--version" => version = Some(flagValue(args, i, "--version")),
                    other => {
                        eprintln!("Unknown release stage flag: {other}");
                        process::exit(1);
                    }
                }
                *i += 1;
            }
            match (manifest, binary, dataDir) {
                (Some(manifest), Some(binary), Some(dataDir)) => Subcommand::ReleaseStage {
                    manifest,
                    binary,
                    dataDir,
                    version,
                },
                _ => {
                    eprintln!(
                        "Usage: zyron-ctl release stage --manifest <file> --binary <file> \
                         --data-dir <dir> [--version <X.Y.Z>]"
                    );
                    process::exit(1);
                }
            }
        }
        "keygen" => {
            let mut out = None;
            while *i < args.len() {
                match args[*i].as_str() {
                    "--out" => out = Some(PathBuf::from(flagValue(args, i, "--out"))),
                    other => {
                        eprintln!("Unknown release keygen flag: {other}");
                        process::exit(1);
                    }
                }
                *i += 1;
            }
            match out {
                Some(out) => Subcommand::ReleaseKeygen { out },
                None => {
                    eprintln!("Usage: zyron-ctl release keygen --out <file>");
                    process::exit(1);
                }
            }
        }
        "sign" => {
            let mut key = None;
            let mut binary = None;
            let mut version = None;
            let mut manifest = None;
            let mut channel = None;
            let mut artifactUrl = None;
            let mut notesUrl = None;
            let mut chain = Vec::new();
            let mut formatBump = false;
            let mut out = None;
            while *i < args.len() {
                match args[*i].as_str() {
                    "--key" => key = Some(PathBuf::from(flagValue(args, i, "--key"))),
                    "--binary" => binary = Some(PathBuf::from(flagValue(args, i, "--binary"))),
                    "--version" => version = Some(flagValue(args, i, "--version")),
                    "--manifest" => {
                        manifest = Some(PathBuf::from(flagValue(args, i, "--manifest")))
                    }
                    "--channel" => channel = Some(flagValue(args, i, "--channel")),
                    "--artifact-url" => artifactUrl = Some(flagValue(args, i, "--artifact-url")),
                    "--notes-url" => notesUrl = Some(flagValue(args, i, "--notes-url")),
                    "--chain" => {
                        chain = flagValue(args, i, "--chain")
                            .split(',')
                            .map(|part| part.trim().to_string())
                            .filter(|part| !part.is_empty())
                            .collect();
                    }
                    "--format-bump" => formatBump = true,
                    "--out" => out = Some(PathBuf::from(flagValue(args, i, "--out"))),
                    other => {
                        eprintln!("Unknown release sign flag: {other}");
                        process::exit(1);
                    }
                }
                *i += 1;
            }
            match (key, binary, version, out) {
                (Some(key), Some(binary), Some(version), Some(out)) => Subcommand::ReleaseSign {
                    key,
                    binary,
                    version,
                    manifest,
                    channel,
                    artifactUrl,
                    notesUrl,
                    chain,
                    formatBump,
                    out,
                },
                _ => {
                    eprintln!(
                        "Usage: zyron-ctl release sign --key <file> --binary <file> --version \
                         <X.Y.Z> --out <file> [--manifest <file>] [--channel <name>] \
                         [--artifact-url <url>] [--notes-url <url>] [--chain <a,b>] \
                         [--format-bump]"
                    );
                    process::exit(1);
                }
            }
        }
        other => {
            eprintln!("Unknown release subcommand: {other}. Use verify, stage, keygen, or sign.");
            process::exit(1);
        }
    }
}

fn executeRemote(flags: &GlobalFlags, sql: &str) -> Result<(), String> {
    let mut client = RemoteClient::connect(&flags.host, flags.port, &flags.user, &flags.database)?;
    let result = client.execute(sql)?;
    let _ = client.close();

    if !result.columns.is_empty() {
        // Compute column widths
        let mut widths: Vec<usize> = result.columns.iter().map(|c| c.len()).collect();
        for row in &result.rows {
            for (col, val) in row.iter().enumerate() {
                if col < widths.len() && val.len() > widths[col] {
                    widths[col] = val.len();
                }
            }
        }

        // Print header
        let header: Vec<String> = result
            .columns
            .iter()
            .enumerate()
            .map(|(idx, name)| format!("{:width$}", name, width = widths[idx]))
            .collect();
        println!(" {} ", header.join(" | "));

        let sep: Vec<String> = widths.iter().map(|w| "-".repeat(*w)).collect();
        println!("-{}-", sep.join("-+-"));

        // Print rows
        for row in &result.rows {
            let formatted: Vec<String> = row
                .iter()
                .enumerate()
                .map(|(idx, val)| {
                    let w = if idx < widths.len() { widths[idx] } else { 0 };
                    format!("{:width$}", val, width = w)
                })
                .collect();
            println!(" {} ", formatted.join(" | "));
        }

        println!("({} rows)", result.rows.len());
    }

    if !result.tag.is_empty() {
        println!("{}", result.tag);
    }

    Ok(())
}

fn handleStatus(flags: &GlobalFlags) -> Result<(), String> {
    println!("Querying server status at {}:{}...", flags.host, flags.port);
    executeRemote(flags, "SELECT * FROM pg_stat_activity")
}

fn handleCheckpoint(flags: &GlobalFlags) -> Result<(), String> {
    println!("Forcing checkpoint on {}:{}...", flags.host, flags.port);
    executeRemote(flags, "CHECKPOINT")
}

fn handleAnalyze(flags: &GlobalFlags, table: &Option<String>) -> Result<(), String> {
    let sql = match table {
        Some(t) => format!("ANALYZE {}", t),
        None => "ANALYZE".to_string(),
    };
    println!("Running {} on {}:{}...", sql, flags.host, flags.port);
    executeRemote(flags, &sql)
}

fn handleVacuum(flags: &GlobalFlags, table: &Option<String>) -> Result<(), String> {
    let sql = match table {
        Some(t) => format!("VACUUM {}", t),
        None => "VACUUM".to_string(),
    };
    println!("Running {} on {}:{}...", sql, flags.host, flags.port);
    executeRemote(flags, &sql)
}

fn handleCompact(flags: &GlobalFlags, table: &Option<String>) -> Result<(), String> {
    let sql = match table {
        Some(t) => format!("OPTIMIZE TABLE {}", t),
        None => "OPTIMIZE TABLE".to_string(),
    };
    println!("Running compaction on {}:{}...", flags.host, flags.port);
    executeRemote(flags, &sql)
}

fn handleBackup(dataDir: &Path, output: &Path) -> Result<(), String> {
    println!("Starting physical backup...");
    println!("  Data directory: {}", dataDir.display());
    println!("  Output: {}", output.display());

    let walDir = dataDir.join("wal");
    let manifest = BackupManager::backup(dataDir, &walDir, output, 0)
        .map_err(|e| format!("backup failed: {}", e))?;

    println!("Backup complete.");
    println!("  Files: {}", manifest.files.len());
    println!("  Created: {}", manifest.createdAt);
    println!(
        "  Manifest written to: {}",
        output.join("manifest.toml").display()
    );
    Ok(())
}

fn handleRestore(input: &Path, dataDir: &Path) -> Result<(), String> {
    println!("Starting restore...");
    println!("  Backup: {}", input.display());
    println!("  Target: {}", dataDir.display());

    RestoreManager::restore(input, dataDir).map_err(|e| format!("restore failed: {}", e))?;

    println!("Restore complete.");
    Ok(())
}

fn handleConfigValidate(configPath: &Path) -> Result<(), String> {
    println!("Validating configuration: {}", configPath.display());

    match ZyronConfig::load(configPath) {
        Ok(config) => {
            println!("Configuration is valid.");
            println!("  Server: {}:{}", config.server.host, config.server.port);
            println!("  Data directory: {}", config.storage.data_dir.display());
            let walDisplay = match &config.wal.wal_dir {
                Some(d) => d.display().to_string(),
                None => "<default: data_dir/wal>".to_string(),
            };
            println!("  WAL directory: {}", walDisplay);
            Ok(())
        }
        Err(e) => {
            eprintln!("Configuration error: {}", e);
            Err(format!("invalid configuration: {}", e))
        }
    }
}

fn handleBenchTpch(flags: &GlobalFlags, scale: f64) -> Result<(), String> {
    let mut client = RemoteClient::connect(&flags.host, flags.port, &flags.user, &flags.database)?;
    let result = bench::runTpch(&mut client, scale);
    let _ = client.close();
    result
}

fn handleBenchTpcc(flags: &GlobalFlags, scale: f64) -> Result<(), String> {
    let mut client = RemoteClient::connect(&flags.host, flags.port, &flags.user, &flags.database)?;
    let result = bench::runTpcc(&mut client, scale);
    let _ = client.close();
    result
}

// ---------------------------------------------------------------------------
// Format substrate handlers
// ---------------------------------------------------------------------------

/// The format registry this binary carries, which the three offline format
/// commands read.
fn formatRegistry() -> Result<&'static zyron_common::format::FormatRegistry, String> {
    zyron_common::format::substrate()
        .map(|substrate| &substrate.formats)
        .map_err(|e| e.to_string())
}

fn handleFormatInspect(file: &Path) -> Result<(), String> {
    let inspection = format_cmd::inspect(file)?;
    println!("{}", inspection.render());
    if !inspection.integrity_ok && inspection.kind.is_some() {
        return Err(format!("{} failed its integrity check", file.display()));
    }
    Ok(())
}

fn handleFormatMigrate(format: &str, to: &Option<String>, path: &Path) -> Result<(), String> {
    let registry = formatRegistry()?;
    let kind = zyron_common::format::FormatKind::from_catalog_name(format).ok_or_else(|| {
        format!(
            "`{format}` is not a format kind. `zyron-ctl release verify` lists the kinds \
             this binary carries"
        )
    })?;
    let target = match to {
        Some(text) => Some(
            text.parse::<zyron_common::format::FormatVersion>()
                .map_err(|e| e.to_string())?,
        ),
        None => None,
    };
    println!(
        "Migrating {kind} files under {} to the current version...",
        path.display()
    );
    let report = format_cmd::migrate(registry, kind, target, path)?;
    print!("{}", report.render());
    if report.failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} file(s) could not be migrated",
            report.failures.len()
        ))
    }
}

fn handleFormatVerify(path: &Path) -> Result<(), String> {
    let registry = formatRegistry()?;
    println!("Verifying every Zyron file under {}...", path.display());
    let report = format_cmd::verify(registry, path)?;
    print!("{}", report.render());
    if report.passed() {
        Ok(())
    } else {
        Err(format!(
            "{} file(s) failed verification",
            report.failures.len()
        ))
    }
}

/// `upgrade check` reads what the running server would do, from the views
/// the gate publishes, rather than guessing from a version number.
fn handleUpgradeCheck(flags: &GlobalFlags, verbose: bool) -> Result<(), String> {
    println!("Upgrade check against {}:{}", flags.host, flags.port);
    println!("\nUpgrade state:");
    executeRemote(flags, format_cmd::statements::SHOW_STATE)?;
    println!("\nFormat registry:");
    executeRemote(flags, format_cmd::statements::FORMAT_REGISTRY)?;
    println!("\nFormat migrations in flight:");
    executeRemote(flags, format_cmd::statements::FORMAT_MIGRATIONS)?;
    println!("\nUser-object rewrites:");
    executeRemote(flags, format_cmd::statements::REWRITES)?;
    if verbose {
        println!("\nDeprecation warnings in the trailing window:");
        executeRemote(flags, format_cmd::statements::DEPRECATION_REPORT)?;
    }
    Ok(())
}

/// `release keygen` draws the key releases are signed with. The private
/// half goes to the file and the public half is printed for the repo
fn handleReleaseKeygen(out: &Path) -> Result<(), String> {
    use zyron_server::upgrade::signing::ReleaseSigningSeed;

    let seed = ReleaseSigningSeed::generate();
    seed.write_to(out).map_err(|e| e.to_string())?;
    println!("release signing key written to {}", out.display());
    println!(
        "Store its contents as the ZYRON_RELEASE_SIGNING_KEY secret of the release workflow and \
         keep a copy where you keep secrets. The public half below goes in \
         crates/zyron-server/release-signing.pub, or in upgrade.release_signing_key on every node \
         when this key signs a feed of your own:"
    );
    println!("{}", seed.verifying_hex());
    Ok(())
}

/// `release sign` signs one binary into a manifest, starting the manifest
/// or adding to the one given
#[allow(clippy::too_many_arguments)]
fn handleReleaseSign(
    key: &Path,
    binary: &Path,
    version: &str,
    manifest: Option<&Path>,
    channel: Option<&str>,
    artifactUrl: Option<&str>,
    notesUrl: Option<&str>,
    chain: &[String],
    formatBump: bool,
    out: &Path,
) -> Result<(), String> {
    use zyron_server::upgrade::feed::{parse_manifest, write_manifest};
    use zyron_server::upgrade::signing::{ReleaseSigningSeed, ReleaseToSign, sign_release};

    let seed = ReleaseSigningSeed::read_from(key).map_err(|e| e.to_string())?;
    let mut manifest: zyron_common::format::ReleaseManifest = match manifest {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("reading {}: {e}", path.display()))?;
            parse_manifest(&text).map_err(|e| e.to_string())?.into()
        }
        None => zyron_common::format::ReleaseManifest {
            channel: channel.unwrap_or("stable").to_string(),
            generated_at_secs: 0,
            releases: Vec::new(),
            signature_scheme: String::new(),
            signature: String::new(),
        },
    };
    if let Some(channel) = channel {
        if manifest.channel != channel {
            return Err(format!(
                "the manifest is for the {} channel, not {channel}",
                manifest.channel
            ));
        }
    }
    let bytes = std::fs::read(binary).map_err(|e| format!("reading {}: {e}", binary.display()))?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let entry = sign_release(
        &seed,
        &mut manifest,
        ReleaseToSign {
            version: version.to_string(),
            artifact_url: artifactUrl.unwrap_or("").to_string(),
            notes_url: notesUrl.unwrap_or("").to_string(),
            upgrade_chain: chain.to_vec(),
            carries_format_bump: formatBump,
            binary: &bytes,
        },
        now,
    )
    .map_err(|e| e.to_string())?;
    write_manifest(&manifest, out).map_err(|e| e.to_string())?;
    println!(
        "signed {} ({} bytes, sha256 {}) into the {} channel manifest at {}",
        entry.version,
        bytes.len(),
        entry.sha256,
        manifest.channel,
        out.display()
    );
    Ok(())
}

fn handleReleaseVerify(today: &Option<String>) -> Result<(), String> {
    let substrate = zyron_common::format::substrate().map_err(|e| e.to_string())?;
    let today = match today {
        Some(date) => date.clone(),
        None => currentIsoDate(),
    };
    println!("Release check for {} against {today}", VERSION);
    let report = zyron_server::release_check::run(substrate, &today);
    println!("{}", report.summary());
    for finding in &report.findings {
        println!("  {finding}");
    }
    let reserved = zyron_server::release_check::reserved_kinds();
    if !reserved.is_empty() {
        let names: Vec<String> = reserved.iter().map(|kind| kind.to_string()).collect();
        println!(
            "  note: {} magic(s) allocated for subsystems that ship later: {}",
            reserved.len(),
            names.join(", ")
        );
    }
    if report.passed() {
        Ok(())
    } else {
        Err(format!(
            "{} release check finding(s)",
            report.findings.len()
        ))
    }
}

/// Today as `YYYY-MM-DD` in UTC, derived from the unix clock so the check
/// needs no calendar dependency.
fn currentIsoDate() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i32;
    let (year, month, day) = zyron_common::interval::ymd_from_days(days);
    format!("{year:04}-{month:02}-{day:02}")
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let (flags, subcmd) = match parseArgs() {
        Some(parsed) => parsed,
        None => return,
    };

    let result = match subcmd {
        Subcommand::Status => handleStatus(&flags),
        Subcommand::Backup {
            ref dataDir,
            ref output,
        } => handleBackup(dataDir, output),
        Subcommand::Restore {
            ref input,
            ref dataDir,
        } => handleRestore(input, dataDir),
        Subcommand::Checkpoint => handleCheckpoint(&flags),
        Subcommand::Analyze { ref table } => handleAnalyze(&flags, table),
        Subcommand::Vacuum { ref table } => handleVacuum(&flags, table),
        Subcommand::Compact { ref table } => handleCompact(&flags, table),
        Subcommand::ConfigValidate { ref config } => handleConfigValidate(config),
        Subcommand::BenchTpch { scale } => handleBenchTpch(&flags, scale),
        Subcommand::BenchTpcc { scale } => handleBenchTpcc(&flags, scale),

        Subcommand::FormatInspect { ref file } => handleFormatInspect(file),
        Subcommand::FormatMigrate {
            ref format,
            ref to,
            ref path,
        } => handleFormatMigrate(format, to, path),
        Subcommand::FormatVerify { ref path } => handleFormatVerify(path),

        Subcommand::UpgradeCheck { verbose } => handleUpgradeCheck(&flags, verbose),
        Subcommand::UpgradeTrigger { ref version } => {
            executeRemote(&flags, &format_cmd::statements::trigger(version))
        }
        Subcommand::UpgradeRollback => executeRemote(&flags, format_cmd::statements::ROLLBACK),
        Subcommand::UpgradePause => executeRemote(&flags, format_cmd::statements::PAUSE),
        Subcommand::UpgradeResume => executeRemote(&flags, format_cmd::statements::RESUME),
        Subcommand::UpgradeShowState => executeRemote(&flags, format_cmd::statements::SHOW_STATE),
        Subcommand::UpgradeAcknowledge { ref category } => {
            executeRemote(&flags, &format_cmd::statements::acknowledge(category))
        }

        Subcommand::DocsGenerate { ref root } => handleDocsGenerate(root),
        Subcommand::DocsCheck { ref root } => handleDocsCheck(root),

        Subcommand::DeprecationReport => {
            executeRemote(&flags, format_cmd::statements::DEPRECATION_REPORT)
        }
        Subcommand::DeprecationGuide { ref item } => {
            executeRemote(&flags, &format_cmd::statements::guide(item))
        }

        Subcommand::ReleaseVerify { ref today } => handleReleaseVerify(today),
        Subcommand::ReleaseStage {
            ref manifest,
            ref binary,
            ref dataDir,
            ref version,
        } => {
            format_cmd::stage_release(manifest, binary, dataDir, version.as_deref()).map(|staged| {
                println!(
                    "staged {} for the {} channel: manifest at {}, binary at {}",
                    staged.version,
                    staged.channel,
                    staged.manifestPath.display(),
                    staged.binaryPath.display()
                );
            })
        }
        Subcommand::ReleaseKeygen { ref out } => handleReleaseKeygen(out),
        Subcommand::ReleaseSign {
            ref key,
            ref binary,
            ref version,
            ref manifest,
            ref channel,
            ref artifactUrl,
            ref notesUrl,
            ref chain,
            formatBump,
            ref out,
        } => handleReleaseSign(
            key,
            binary,
            version,
            manifest.as_deref(),
            channel.as_deref(),
            artifactUrl.as_deref(),
            notesUrl.as_deref(),
            chain,
            formatBump,
            out,
        ),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// SQL statement reference
// ---------------------------------------------------------------------------

/// The default reference root, relative to the repository.
const DOCS_REFERENCE_ROOT: &str = "docs/business/sql/reference";

/// Reads `docs generate` and `docs check`.
fn parseDocs(args: &[String], i: &mut usize) -> Subcommand {
    // The subcommand word itself is still under the cursor
    *i += 1;
    if *i >= args.len() {
        eprintln!("Usage: zyron-ctl docs <generate|check> [--root <path>]");
        process::exit(1);
    }
    let action = args[*i].clone();
    *i += 1;
    let mut root = PathBuf::from(DOCS_REFERENCE_ROOT);
    while *i < args.len() {
        match args[*i].as_str() {
            "--root" => root = PathBuf::from(flagValue(args, i, "--root")),
            _ => break,
        }
        *i += 1;
    }
    match action.as_str() {
        "generate" => Subcommand::DocsGenerate { root },
        "check" => Subcommand::DocsCheck { root },
        other => {
            eprintln!("Unknown docs action: {other}. Use generate or check.");
            process::exit(1);
        }
    }
}

/// Writes the reference tree from the grammar registry.
fn handleDocsGenerate(root: &Path) -> Result<(), String> {
    let count = docs_cmd::generate(root)
        .map_err(|e| format!("writing the reference to {} failed: {e}", root.display()))?;
    println!("Wrote {} page(s) to {}", count, root.display());
    Ok(())
}

/// Compares the tree on disk against what the registry writes now.
fn handleDocsCheck(root: &Path) -> Result<(), String> {
    let findings = docs_cmd::check(root)
        .map_err(|e| format!("reading the reference under {} failed: {e}", root.display()))?;
    if findings.is_empty() {
        println!(
            "The reference under {} is what the registry writes",
            root.display()
        );
        return Ok(());
    }
    let mut message = format!(
        "the reference under {} is not what the registry writes:",
        root.display()
    );
    for finding in &findings {
        message.push_str(
            "
  ",
        );
        message.push_str(finding);
    }
    message.push_str(
        "
Run `zyron-ctl docs generate` to rewrite it.",
    );
    Err(message)
}
