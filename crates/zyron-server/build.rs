//! Hands the target triple to the crate.
//!
//! A release manifest is published per target, because a Linux node and a
//! Windows node download different binaries, and the node polls the
//! manifest named after the target it was built for. Cargo tells a build
//! script the target and nothing else, so it is passed on here as an
//! environment variable the crate reads at compile time

fn main() {
    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=ZYRON_TARGET={target}");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=release-signing.pub");
}
