#![allow(non_snake_case)]
//! Zyron cluster management and admin tool.
//!
//! Provides subcommands for server status, backup, restore, checkpoint,
//! analyze, vacuum, compaction, configuration validation, and benchmarks.

mod bench;
mod format_cmd;
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

    DeprecationReport,
    DeprecationGuide {
        item: String,
    },

    ReleaseVerify {
        today: Option<String>,
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
  deprecation report                        List deprecation warnings in the trailing window
  deprecation guides <item>                 Print one item's migration guide

Release subcommand (checks this binary):
  release verify [--today <YYYY-MM-DD>]     Check every format bump, fixture, and retirement

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
        other => {
            eprintln!(
                "Unknown upgrade subcommand: {other}. Use check, trigger, rollback, pause, \
                 resume, or show-state."
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
    if *i >= args.len() || args[*i] != "verify" {
        eprintln!("Usage: zyron-ctl release verify [--today <YYYY-MM-DD>]");
        process::exit(1);
    }
    *i += 1;
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

        Subcommand::DeprecationReport => {
            executeRemote(&flags, format_cmd::statements::DEPRECATION_REPORT)
        }
        Subcommand::DeprecationGuide { ref item } => {
            executeRemote(&flags, &format_cmd::statements::guide(item))
        }

        Subcommand::ReleaseVerify { ref today } => handleReleaseVerify(today),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}
