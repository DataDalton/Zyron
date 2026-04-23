// -----------------------------------------------------------------------------
// Runtime router registration trait.
//
// Decouples the wire DDL dispatcher and the server startup path from the
// concrete HTTP gateway router. DDL handlers call register/unregister on the
// trait object held in ServerState so newly created endpoints become routable
// without a restart and dropped endpoints stop serving traffic immediately.
// -----------------------------------------------------------------------------

use async_trait::async_trait;

use zyron_catalog::{EndpointEntry, EndpointId};
use zyron_common::Result;

/// Hooks invoked by DDL handlers and startup recovery to keep the live HTTP
/// router aligned with catalog state.
#[async_trait]
pub trait EndpointRegistrar: Send + Sync {
    /// Installs or replaces the compiled route for the given catalog entry.
    async fn register(&self, entry: &EndpointEntry) -> Result<()>;

    /// Removes the compiled route tied to the given endpoint id.
    async fn unregister(&self, endpoint_id: EndpointId) -> Result<()>;

    /// Enables or disables the route. Disabling removes the route from the
    /// lookup list, enabling rebuilds it from the supplied catalog entry.
    async fn set_enabled(&self, entry: &EndpointEntry, enabled: bool) -> Result<()>;
}
