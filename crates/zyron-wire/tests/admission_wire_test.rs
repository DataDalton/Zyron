//! Admission as a client experiences it, over a real socket.
//!
//! Every other test of the controller reaches into it directly. This one does
//! not: it stands up the server, connects over the PostgreSQL wire protocol,
//! and asks what a driver would see. Two things can only be checked from here.
//!
//! The first is that admission is actually on the serving path. A controller
//! that decides correctly and is never consulted is a controller that does
//! nothing, and no unit test of the controller can tell the difference.
//!
//! The second is the framing. A node refusing work because it is at capacity
//! is not a failed query: nothing is wrong with it, and the caller should
//! retry or go elsewhere. That distinction only survives to the client if the
//! refusal arrives as SQLSTATE 53400 with the reason attached, so that is what
//! is asserted rather than the presence of some error.
//!
//! Run: cargo test -p zyron-wire --test admission_wire_test
//!
//! One process, one controller, and these tests move its counters, so they run
//! one at a time.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::create_test_server;
use zyron_pressure::capability::{OperatorCoefficients, OperatorKind};
use zyron_pressure::pressure::WorkloadClass;
use zyron_pressure::pressure_control::PressureController;
use zyron_wire::pg_client::{ClientConfig, PgClient};

/// The process controller is shared, so a test that saturates a class would
/// otherwise refuse the queries another test expects to be admitted.
static SERIALIZE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Makes every plan price in seconds.
///
/// Without this the test table is small enough that any query over it costs
/// microseconds, lands below the bypass threshold, and never reaches the
/// admission decision this test exists to exercise. Building a table large
/// enough to cost a real second would make the test slow for no extra
/// coverage: what is under test is the decision and its framing, not the
/// pricing, which has its own tests.
fn price_every_query_in_seconds() {
    let mut expensive = OperatorCoefficients::cold_start();
    for kind in OperatorKind::ALL {
        // Ten milliseconds a row, so even a thousand-row estimate is bulk work
        expensive.ns_per_unit[kind.index()] = 10_000_000.0;
        expensive.sample_count[kind.index()] = 10_000;
    }
    PressureController::global().coefficients().seed(&expensive);
}

/// Restores the cost model so a later test in this binary is not priced
/// against the exaggerated one.
fn restore_pricing() {
    PressureController::global()
        .coefficients()
        .seed(&OperatorCoefficients::cold_start());
}

/// Fills a class to its ceiling and its queue, which is the state in which the
/// next arrival is refused.
///
/// Done through the counters rather than by issuing real queries, because
/// reaching a bulk queue depth of two hundred and fifty six with real
/// five-second queries would take minutes and prove the same thing.
struct SaturatedClass {
    class: WorkloadClass,
    running: u32,
    queued: u32,
}

impl SaturatedClass {
    fn hold(class: WorkloadClass, work_seconds: f64) -> Self {
        let controller = PressureController::global();
        let counters = controller.counters(class);
        let ceiling = controller.ceiling(class);
        for _ in 0..ceiling {
            counters.start_direct(work_seconds);
        }
        let depth = class.queue_depth() as u32;
        for _ in 0..depth {
            counters.enqueue(work_seconds);
        }
        Self {
            class,
            running: ceiling,
            queued: depth,
        }
    }
}

impl Drop for SaturatedClass {
    fn drop(&mut self) {
        let counters = PressureController::global().counters(self.class);
        for _ in 0..self.running {
            counters.complete(0.0, 0.0);
        }
        for _ in 0..self.queued {
            counters.abandon_queued(0.0);
        }
    }
}

async fn reserve_port() -> u16 {
    let probe = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind probe");
    let port = probe.local_addr().expect("addr").port();
    drop(probe);
    port
}

async fn wait_until_listening(port: u16) {
    for _ in 0..200 {
        if tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .is_ok()
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("server never began listening on port {port}");
}

/// Stands up the server on a real socket and returns the port.
async fn serve(state: Arc<zyron_wire::connection::ServerState>) -> u16 {
    let port = reserve_port().await;
    let mut config = zyron_common::config::ServerConfig::default();
    config.host = "127.0.0.1".to_string();
    config.port = port;
    config.quic_enabled = false;
    let leaked: &'static zyron_common::config::ServerConfig = Box::leak(Box::new(config));
    tokio::spawn(async move {
        let _ = zyron_wire::start_server(leaked, state).await;
    });
    wait_until_listening(port).await;
    port
}

async fn connect(port: u16) -> PgClient {
    let cfg = ClientConfig {
        user: "zyron".to_string(),
        database: "zyron".to_string(),
        application_name: "admission-test".to_string(),
        password: None,
        connect_timeout: Duration::from_secs(10),
        statement_timeout: Duration::from_secs(20),
    };
    let addr: std::net::SocketAddr = ([127, 0, 0, 1], port).into();
    PgClient::connect(addr, &cfg).await.expect("handshake")
}

/// A node with capacity admits, a node without refuses with the code that says
/// so, and the same node admits again once the pressure clears.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_shed_reaches_the_client_as_a_capacity_refusal() {
    let _guard = SERIALIZE.lock().unwrap_or_else(|e| e.into_inner());
    let (state, _schema, _tmp) = create_test_server().await;
    common::exec_ddl(
        &state,
        &mut common::new_session(),
        "CREATE TABLE orders (id BIGINT, amount BIGINT)",
    )
    .await
    .expect("ddl");
    for i in 0..64 {
        common::exec_dml(
            &state,
            &format!("INSERT INTO zyron_test.orders VALUES ({i}, {})", i * 10),
        )
        .await;
    }

    let port = serve(state).await;
    let mut client = connect(port).await;

    // With the node quiet, the query runs
    let admitted = client
        .simple_query("SELECT id FROM zyron_test.orders")
        .await
        .expect("a quiet node admitted the query");
    assert!(!admitted.is_empty(), "the admitted query returned nothing");

    price_every_query_in_seconds();
    let refusal = {
        let _full = SaturatedClass::hold(WorkloadClass::Bulk, 5.0);
        client
            .simple_query("SELECT id FROM zyron_test.orders")
            .await
            .expect_err("a full node admitted a query it had no capacity for")
    };

    assert_eq!(
        refusal.sqlstate(),
        Some("53400"),
        "a capacity refusal was framed as {refusal:?}"
    );
    assert!(
        refusal.is_admission_shed(),
        "a client cannot tell this refusal apart from a failure: {refusal:?}"
    );
    let text = refusal.to_string();
    assert!(
        text.contains("shedding") || text.contains("admission refused"),
        "the refusal did not say what happened: {text}"
    );

    // The pressure is gone with the guard, and the same connection works again
    restore_pricing();
    let after = client
        .simple_query("SELECT id FROM zyron_test.orders")
        .await
        .expect("the node stayed refusing after the pressure cleared");
    assert!(!after.is_empty());
}

/// A refusal must not take the connection down with it. The caller is being
/// asked to retry, which it cannot do on a session the server killed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_refused_query_leaves_the_session_usable() {
    let _guard = SERIALIZE.lock().unwrap_or_else(|e| e.into_inner());
    let (state, _schema, _tmp) = create_test_server().await;
    common::exec_ddl(
        &state,
        &mut common::new_session(),
        "CREATE TABLE items (id BIGINT)",
    )
    .await
    .expect("ddl");
    common::exec_dml(&state, "INSERT INTO zyron_test.items VALUES (1)").await;

    let port = serve(state).await;
    let mut client = connect(port).await;

    price_every_query_in_seconds();
    {
        let _full = SaturatedClass::hold(WorkloadClass::Bulk, 5.0);
        for attempt in 0..3 {
            let refusal = client
                .simple_query("SELECT id FROM zyron_test.items")
                .await
                .expect_err("attempt {attempt} was admitted on a full node");
            assert_eq!(
                refusal.sqlstate(),
                Some("53400"),
                "attempt {attempt} was refused as {refusal:?}"
            );
        }
    }
    restore_pricing();

    // Three refusals later the session still works
    let rows = client
        .simple_query("SELECT id FROM zyron_test.items")
        .await
        .expect("the session did not survive being refused");
    assert!(!rows.is_empty());
}

/// A query cheap enough to bypass admission runs while the node is refusing
/// everything else, which is what keeps a point lookup fast under load.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_cheap_query_still_runs_on_a_shedding_node() {
    let _guard = SERIALIZE.lock().unwrap_or_else(|e| e.into_inner());
    let (state, _schema, _tmp) = create_test_server().await;
    common::exec_ddl(
        &state,
        &mut common::new_session(),
        "CREATE TABLE tiny (id BIGINT)",
    )
    .await
    .expect("ddl");
    common::exec_dml(&state, "INSERT INTO zyron_test.tiny VALUES (7)").await;

    let port = serve(state).await;
    let mut client = connect(port).await;

    // Every class held at its queue depth, so nothing that reaches the
    // decision can be admitted
    let _bulk = SaturatedClass::hold(WorkloadClass::Bulk, 5.0);
    let _interactive = SaturatedClass::hold(WorkloadClass::Interactive, 0.05);

    // Priced on the real cost model, this costs microseconds and is below the
    // bypass threshold
    let rows = client
        .simple_query("SELECT id FROM zyron_test.tiny")
        .await
        .expect("a query below the bypass threshold was refused");
    assert!(!rows.is_empty());
}

/// Everything the wire refuses is also visible in the views, so an operator
/// asking why a client saw 53400 gets the same answer the client did.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_refusal_is_recorded_where_an_operator_can_see_it() {
    let _guard = SERIALIZE.lock().unwrap_or_else(|e| e.into_inner());
    let (state, _schema, _tmp) = create_test_server().await;
    common::exec_ddl(
        &state,
        &mut common::new_session(),
        "CREATE TABLE audit_src (id BIGINT)",
    )
    .await
    .expect("ddl");
    common::exec_dml(&state, "INSERT INTO zyron_test.audit_src VALUES (1)").await;

    let port = serve(state).await;
    let mut client = connect(port).await;
    let controller = PressureController::global();
    let before = controller.counters(WorkloadClass::Bulk).shed_total();

    price_every_query_in_seconds();
    {
        let _full = SaturatedClass::hold(WorkloadClass::Bulk, 5.0);
        client
            .simple_query("SELECT id FROM zyron_test.audit_src")
            .await
            .expect_err("expected a refusal");
    }
    restore_pricing();

    assert!(
        controller.counters(WorkloadClass::Bulk).shed_total() > before,
        "the wire refused a query the counters do not know about"
    );
    let recent = controller.admissions(16);
    assert!(
        recent.iter().any(|r| r.decision == "shed"),
        "no refusal reached zyron_sys.pressure.admissions"
    );
}
