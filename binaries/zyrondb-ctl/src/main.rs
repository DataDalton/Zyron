#![allow(non_snake_case)]
//! ZyronDB cluster management and admin tool.
//!
//! Provides subcommands for server status, backup, restore, checkpoint,
//! analyze, vacuum, compaction, configuration validation, and benchmarks.

mod bench;
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
    Backup { dataDir: PathBuf, output: PathBuf },
    Restore { input: PathBuf, dataDir: PathBuf },
    Checkpoint,
    Analyze { table: Option<String> },
    Vacuum { table: Option<String> },
    Compact { table: Option<String> },
    ConfigValidate { config: PathBuf },
    BenchTpch { scale: f64 },
    BenchTpcc { scale: f64 },
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

fn printHelp() {
    println!(
        "zyrondb-ctl {} - ZyronDB cluster management tool

Usage: zyrondb-ctl [global-flags] <subcommand> [subcommand-flags]

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
                println!("zyrondb-ctl {}", VERSION);
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
                eprintln!("Usage: zyrondb-ctl config validate --config <path>");
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
                eprintln!("Usage: zyrondb-ctl bench <tpch|tpcc> --scale <N>");
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
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}
