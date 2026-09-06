//! Zyron server entry point.

use zyron_server::RunOutcome;

fn main() {
    let opts = match zyron_server::parse_cli_args() {
        Some(opts) => opts,
        None => return, // --help, --version, or --capabilities was printed
    };

    let config =
        match zyron_server::config::ZyronConfig::load_with_overrides(opts.config_path.as_deref()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Failed to load configuration: {}", e);
                std::process::exit(1);
            }
        };

    // The runtime is built after the config loads so server.worker_threads
    // sizes it. Zero means the tokio default, one thread per core
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all();
    if config.server.worker_threads > 0 {
        builder.worker_threads(config.server.worker_threads);
    }
    let runtime = match builder.build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Failed to build async runtime: {}", e);
            std::process::exit(1);
        }
    };

    let outcome = runtime.block_on(async move {
        let server = match zyron_server::Server::init(config, &opts).await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to initialize server: {}", e);
                std::process::exit(1);
            }
        };

        match server.run().await {
            Ok(outcome) => outcome,
            Err(e) => {
                eprintln!("Server error: {}", e);
                std::process::exit(1);
            }
        }
    });

    // The runtime is gone before the process is replaced, so no worker
    // thread is left holding a file the new process opens
    drop(runtime);
    if let RunOutcome::Restart { binary, args } = outcome {
        restart(&binary, &args);
    }
}

/// Starts the binary at the live path in this process's place.
///
/// On Unix the process image is replaced, so the service manager keeps the
/// same PID and the new binary inherits the descriptors. On Windows a
/// process cannot replace itself, so the new one is started detached and
/// this one exits once it has
#[cfg(unix)]
fn restart(binary: &std::path::Path, args: &[String]) {
    use std::os::unix::process::CommandExt;
    let err = std::process::Command::new(binary).args(args).exec();
    eprintln!("could not start {} in place: {err}", binary.display());
    std::process::exit(1);
}

#[cfg(not(unix))]
fn restart(binary: &std::path::Path, args: &[String]) {
    match std::process::Command::new(binary).args(args).spawn() {
        Ok(child) => {
            eprintln!(
                "started {} as process {}, this process exits",
                binary.display(),
                child.id()
            );
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("could not start {}: {e}", binary.display());
            std::process::exit(1);
        }
    }
}
