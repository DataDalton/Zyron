//! Self contained media engine for Zyron
//!
//! Covers media descriptors with a versioned binary codec, a content
//! addressed object store with refcounts and lz4 compression, image
//! metadata and transformations, pure Rust video, audio and document
//! metadata parsers, document text extraction and markdown conversion,
//! presigned handles, external references through opendal, and tool
//! backed operations that invoke configured ffmpeg or tesseract binaries
//! and error actionably when they are absent. Database wiring such as SQL
//! types and functions lives outside this crate

pub mod audio_meta;
pub mod descriptor;
pub mod document;
pub mod error;
pub mod external;
pub mod format;
pub mod image_meta;
pub mod image_ops;
pub mod presign;
pub mod skeleton;
pub mod store;
pub mod video_meta;

pub use audio_meta::audio_metadata;
pub use descriptor::{MediaDescriptor, MediaKind, StorageMode, is_descriptor};
pub use document::{
    document_extract_text, document_metadata, document_page_count, document_to_markdown,
};
pub use error::{MediaError, MediaResult};
pub use external::ExternalFetcher;
pub use image_meta::{ImageFormatKind, detect_image_format, image_metadata};
pub use image_ops::{ResizeMode, convert_format, crop, resize, rotate};
pub use presign::{VerifiedHandle, sign, verify};
pub use skeleton::{
    MediaToolConfig, audio_transcode, audio_transcribe, audio_trim, image_embed, image_ocr,
    video_extract_audio, video_extract_frame, video_thumbnail, video_transcode,
};
pub use store::{MediaStore, StoredObject};
pub use video_meta::video_metadata;
