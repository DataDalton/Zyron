//! Zyron server entry point.

fn main() {
    let opts = match zyron_server::parse_cli_args() {
        Some(opts) => opts,
        None => return, // --help or --version was printed
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

    runtime.block_on(async move {
        let server = match zyron_server::Server::init(config, &opts).await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to initialize server: {}", e);
                std::process::exit(1);
            }
        };

        if let Err(e) = server.run().await {
            eprintln!("Server error: {}", e);
            std::process::exit(1);
        }
    });
}
