//! Error types for the media engine

use thiserror::Error;

/// Result alias for media engine operations
pub type MediaResult<T> = std::result::Result<T, MediaError>;

/// Errors produced by the media engine
#[derive(Debug, Error)]
pub enum MediaError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("corrupt media descriptor: {0}")]
    CorruptDescriptor(String),

    #[error("corrupt media object: {0}")]
    CorruptObject(String),

    #[error("media object not found in store: {0}")]
    ObjectNotFound(String),

    #[error(
        "media object integrity check failed, address is {expected} but content hashes to {actual}"
    )]
    IntegrityMismatch { expected: String, actual: String },

    #[error("release of unreferenced media object {0}")]
    ReleaseUnreferenced(String),

    #[error("{0}")]
    UnsupportedFormat(String),

    #[error("{0}")]
    InvalidArgument(String),

    #[error("{operation} requires {tool}, set {config_key} in zyron.toml to the {tool} binary")]
    ToolMissing {
        operation: String,
        tool: String,
        config_key: String,
    },

    #[error(
        "{config_key} is set to {path} but no file exists there, point {config_key} in zyron.toml at the {tool} binary"
    )]
    ToolPathInvalid {
        tool: String,
        config_key: String,
        path: String,
    },

    #[error("{tool} failed for {operation} with {status}: {stderr}")]
    ToolFailed {
        tool: String,
        operation: String,
        status: String,
        stderr: String,
    },

    #[error(
        "{operation} requires a registered model and none is registered, register a model in the model registry first"
    )]
    ModelMissing { operation: String },

    #[error(
        "{operation} requires a model inference runtime, this build links no inference runtime for registered models"
    )]
    ModelRuntimeUnavailable { operation: String },

    #[error("presigned handle invalid: {0}")]
    InvalidHandle(String),

    #[error("presigned handle expired at {expires_at_unix}, current time is {now_unix}")]
    HandleExpired { expires_at_unix: i64, now_unix: i64 },

    #[error(
        "unsupported external uri scheme in {uri}, supported schemes are file, s3, gcs, azblob, http and https"
    )]
    UnsupportedScheme { uri: String },

    #[error("external media error: {0}")]
    External(String),
}

impl From<MediaError> for zyron_common::ZyronError {
    fn from(err: MediaError) -> Self {
        zyron_common::ZyronError::ExecutionError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_to_zyron_error_with_full_message() {
        let err = MediaError::ToolMissing {
            operation: "video transcoding".to_string(),
            tool: "ffmpeg".to_string(),
            config_key: "media.ffmpeg_path".to_string(),
        };
        let msg = err.to_string();
        let zerr: zyron_common::ZyronError = err.into();
        assert_eq!(zerr.to_string(), format!("Execution error: {msg}"));
        assert!(msg.contains("media.ffmpeg_path"));
    }
}
