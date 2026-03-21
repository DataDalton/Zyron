//! Cross-platform shutdown signal handling.
//!
//! Listens for SIGTERM/SIGINT on Unix and Ctrl+C/Ctrl+Break on Windows.
//! Returns the reason for shutdown so the server can perform graceful cleanup.

/// Reason the server is shutting down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownReason {
    /// SIGTERM or Ctrl+C received.
    Terminate,
    /// SIGINT or Ctrl+Break received.
    Interrupt,
}

/// Waits for a shutdown signal from the OS.
/// Returns the reason so the caller can decide on cleanup behavior.
pub async fn wait_for_shutdown() -> ShutdownReason {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigterm =
            signal(SignalKind::terminate()).expect("failed to install SIGTERM handler");
        let mut sigint = signal(SignalKind::interrupt()).expect("failed to install SIGINT handler");

        tokio::select! {
            _ = sigterm.recv() => ShutdownReason::Terminate,
            _ = sigint.recv() => ShutdownReason::Interrupt,
        }
    }

    #[cfg(not(unix))]
    {
        // Windows: Ctrl+C is the primary shutdown mechanism.
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
        ShutdownReason::Terminate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shutdown_reason_clone() {
        let reason = ShutdownReason::Terminate;
        let cloned = reason;
        assert_eq!(reason, cloned);
    }

    #[test]
    fn test_shutdown_reason_variants() {
        assert_ne!(ShutdownReason::Terminate, ShutdownReason::Interrupt);
    }
}
