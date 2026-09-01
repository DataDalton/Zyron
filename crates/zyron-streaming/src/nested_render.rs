//! The one codec for spelling a stored STRUCT or MAP as json text.
//!
//! A binary sink writes the stored layout through unchanged, but a text
//! format has to spell it, and a nested leaf is spelled by the same scalar
//! formatting every other stored value of that type uses. That formatting
//! lives in the executor, which sits above this crate, so the engine
//! registers its own routines here during startup rather than this crate
//! growing a second copy that could drift from the first and give one value
//! two spellings.
//!
//! Nothing here decodes anything itself. Without a registration a nested
//! value stays refused by the text formats, which is what happens in a
//! process that never starts an engine.

use std::sync::OnceLock;

use zyron_catalog::schema::NestedShape;
use zyron_common::{Result, ZyronError};

/// Renders a stored nested value as the json text a reader sees.
pub type RenderFn = fn(&[u8], &NestedShape) -> String;

/// Encodes json text into the stored layout a declared shape fixes. The third
/// argument is the column name, which the encoder names in its errors.
pub type ParseFn = fn(&str, &NestedShape, &str) -> Result<Vec<u8>>;

struct NestedCodec {
    render: RenderFn,
    parse: ParseFn,
}

static CODEC: OnceLock<NestedCodec> = OnceLock::new();

/// Registers the engine's nested codec, once per process. A later call is
/// ignored, so a test that registers cannot disturb an engine already running
/// in the same process.
pub fn register(render: RenderFn, parse: ParseFn) {
    let _ = CODEC.set(NestedCodec { render, parse });
}

/// True when a nested value can be spelled as text in this process.
pub fn is_registered() -> bool {
    CODEC.get().is_some()
}

/// Renders a stored nested value as json text.
pub fn render(bytes: &[u8], shape: &NestedShape) -> Result<String> {
    match CODEC.get() {
        Some(codec) => Ok((codec.render)(bytes, shape)),
        None => Err(ZyronError::StreamingError(
            "no nested codec is registered, so a STRUCT or MAP cannot be written as text"
                .to_string(),
        )),
    }
}

/// Encodes json text into the stored layout for a declared shape.
pub fn parse(text: &str, shape: &NestedShape, column: &str) -> Result<Vec<u8>> {
    match CODEC.get() {
        Some(codec) => (codec.parse)(text, shape, column),
        None => Err(ZyronError::StreamingError(format!(
            "no nested codec is registered, so column {column} cannot be read from text"
        ))),
    }
}
