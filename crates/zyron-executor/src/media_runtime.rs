//! Process wide media runtime state.
//!
//! The media store itself travels on the ExecutionContext like every other
//! per server handle. What lives here is the state the scalar function
//! bridge needs without a context: the presigned URL secret, the external
//! tool configuration, and the shared external object fetcher. The server
//! installs them at startup; unconfigured tools answer with the actionable
//! errors zyron-media raises.

use std::sync::OnceLock;
use std::time::Duration;

use parking_lot::RwLock;

use zyron_media::skeleton::MediaToolConfig;

static PRESIGN_SECRET: OnceLock<[u8; 32]> = OnceLock::new();
static TOOL_CONFIG: RwLock<Option<MediaToolConfig>> = RwLock::new(None);
static FETCHER: OnceLock<zyron_media::external::ExternalFetcher> = OnceLock::new();

/// Installs the secret presigned URLs sign with. First install wins, which
/// keeps issued URLs verifiable for the process lifetime
pub fn install_presign_secret(secret: [u8; 32]) {
    let _ = PRESIGN_SECRET.set(secret);
}

/// The signing secret, or a clear error when the server never installed one
pub fn presign_secret() -> zyron_common::Result<&'static [u8; 32]> {
    PRESIGN_SECRET.get().ok_or_else(|| {
        zyron_common::ZyronError::ExecutionError(
            "presigned URLs need the server's signing secret, which only a running server installs"
                .to_string(),
        )
    })
}

/// Installs the external tool paths media operations invoke
pub fn install_tool_config(config: MediaToolConfig) {
    *TOOL_CONFIG.write() = Some(config);
}

/// The current tool configuration, empty when the server installed none
pub fn tool_config() -> MediaToolConfig {
    TOOL_CONFIG.read().clone().unwrap_or(MediaToolConfig {
        ffmpeg_path: None,
        tesseract_path: None,
    })
}

/// The shared fetcher external references resolve through, with a one
/// minute content cache
pub fn external_fetcher() -> &'static zyron_media::external::ExternalFetcher {
    FETCHER.get_or_init(|| zyron_media::external::ExternalFetcher::new(Duration::from_secs(60)))
}
