//! ZyronDB server entry point.

#[tokio::main]
async fn main() {
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
}
