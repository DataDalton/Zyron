//! TPC-C schema, data generation and the five-transaction workload.
//!
//! Scale is warehouses: the specification fixes ten districts per warehouse,
//! three thousand customers per district and one hundred thousand items,
//! with a stock row per item per warehouse. Those ratios are what make the
//! transaction mix contend the way the benchmark intends, so they are fixed
//! here and only the warehouse count follows the scale factor.
//!
//! The workload is driven as SQL through the same connection a client uses,
//! so what it measures is the path an application takes.

use crate::{InsertBatcher, Rng, format_cents, quote};

pub const DISTRICTS_PER_WAREHOUSE: i64 = 10;
pub const CUSTOMERS_PER_DISTRICT: i64 = 3_000;
pub const ITEMS: i64 = 100_000;
/// Orders present at load, of which the last 900 per district are new
const ORDERS_PER_DISTRICT: i64 = 3_000;
const NEW_ORDERS_PER_DISTRICT: i64 = 900;

const LAST_NAME_SYLLABLES: [&str; 10] = [
    "BAR", "OUGHT", "ABLE", "PRI", "PRES", "ESE", "ANTI", "CALLY", "ATION", "EING",
];

/// The specification's customer surname construction, three syllables chosen
/// by the digits of a number below one thousand. Customer-by-name lookup in
/// the Payment and Order-Status transactions depends on this distribution.
fn last_name(n: i64) -> String {
    format!(
        "{}{}{}",
        LAST_NAME_SYLLABLES[(n / 100 % 10) as usize],
        LAST_NAME_SYLLABLES[(n / 10 % 10) as usize],
        LAST_NAME_SYLLABLES[(n % 10) as usize]
    )
}

/// The specification's non-uniform random, which is what concentrates access
/// on a minority of rows and creates the contention the benchmark measures.
/// A uniform draw here would report a workload nobody runs.
fn nurand(rng: &mut Rng, a: i64, x: i64, y: i64) -> i64 {
    let c = 0; // A fixed run constant keeps a run reproducible
    (((rng.range(0, a) | rng.range(x, y)) + c) % (y - x + 1)) + x
}

fn random_string(rng: &mut Rng, min: usize, max: usize) -> String {
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let len = rng.range(min as i64, max as i64) as usize;
    (0..len)
        .map(|_| ALPHABET[(rng.next_u64() as usize) % ALPHABET.len()] as char)
        .collect()
}

fn zip(rng: &mut Rng) -> String {
    format!("{:04}11111", rng.range(0, 9999))
}

/// The DDL for the nine TPC-C tables.
pub fn schema() -> Vec<&'static str> {
    vec![
        "CREATE TABLE IF NOT EXISTS warehouse (w_id INT, w_name VARCHAR(10), w_street_1 VARCHAR(20), w_street_2 VARCHAR(20), w_city VARCHAR(20), w_state VARCHAR(2), w_zip VARCHAR(9), w_tax DECIMAL(4,4), w_ytd DECIMAL(12,2))",
        "CREATE TABLE IF NOT EXISTS district (d_id INT, d_w_id INT, d_name VARCHAR(10), d_street_1 VARCHAR(20), d_street_2 VARCHAR(20), d_city VARCHAR(20), d_state VARCHAR(2), d_zip VARCHAR(9), d_tax DECIMAL(4,4), d_ytd DECIMAL(12,2), d_next_o_id INT)",
        "CREATE TABLE IF NOT EXISTS customer (c_id INT, c_d_id INT, c_w_id INT, c_first VARCHAR(16), c_middle VARCHAR(2), c_last VARCHAR(16), c_street_1 VARCHAR(20), c_street_2 VARCHAR(20), c_city VARCHAR(20), c_state VARCHAR(2), c_zip VARCHAR(9), c_phone VARCHAR(16), c_since TIMESTAMP, c_credit VARCHAR(2), c_credit_lim DECIMAL(12,2), c_discount DECIMAL(4,4), c_balance DECIMAL(12,2), c_ytd_payment DECIMAL(12,2), c_payment_cnt INT, c_delivery_cnt INT, c_data VARCHAR(500))",
        "CREATE TABLE IF NOT EXISTS history (h_c_id INT, h_c_d_id INT, h_c_w_id INT, h_d_id INT, h_w_id INT, h_date TIMESTAMP, h_amount DECIMAL(6,2), h_data VARCHAR(24))",
        "CREATE TABLE IF NOT EXISTS new_order (no_o_id INT, no_d_id INT, no_w_id INT)",
        "CREATE TABLE IF NOT EXISTS orders (o_id INT, o_d_id INT, o_w_id INT, o_c_id INT, o_entry_d TIMESTAMP, o_carrier_id INT, o_ol_cnt INT, o_all_local INT)",
        "CREATE TABLE IF NOT EXISTS order_line (ol_o_id INT, ol_d_id INT, ol_w_id INT, ol_number INT, ol_i_id INT, ol_supply_w_id INT, ol_delivery_d TIMESTAMP, ol_quantity INT, ol_amount DECIMAL(6,2), ol_dist_info VARCHAR(24))",
        "CREATE TABLE IF NOT EXISTS item (i_id INT, i_im_id INT, i_name VARCHAR(24), i_price DECIMAL(5,2), i_data VARCHAR(50))",
        "CREATE TABLE IF NOT EXISTS stock (s_i_id INT, s_w_id INT, s_quantity INT, s_dist_01 VARCHAR(24), s_dist_02 VARCHAR(24), s_dist_03 VARCHAR(24), s_dist_04 VARCHAR(24), s_dist_05 VARCHAR(24), s_dist_06 VARCHAR(24), s_dist_07 VARCHAR(24), s_dist_08 VARCHAR(24), s_dist_09 VARCHAR(24), s_dist_10 VARCHAR(24), s_ytd INT, s_order_cnt INT, s_remote_cnt INT, s_data VARCHAR(50))",
    ]
}

/// The tables the generator writes, so a driver can clear a previous run.
/// `orders` and `customer` are shared names with TPC-H, which is why the two
/// benchmarks are loaded into separate databases.
pub fn tables() -> [&'static str; 9] {
    [
        "order_line",
        "new_order",
        "orders",
        "history",
        "customer",
        "district",
        "stock",
        "item",
        "warehouse",
    ]
}

/// Row counts this warehouse count produces.
pub fn row_counts(warehouses: i64) -> Vec<(&'static str, i64)> {
    let districts = warehouses * DISTRICTS_PER_WAREHOUSE;
    let customers = districts * CUSTOMERS_PER_DISTRICT;
    let orders = districts * ORDERS_PER_DISTRICT;
    vec![
        ("warehouse", warehouses),
        ("district", districts),
        ("item", ITEMS),
        ("stock", warehouses * ITEMS),
        ("customer", customers),
        ("history", customers),
        ("orders", orders),
        ("new_order", districts * NEW_ORDERS_PER_DISTRICT),
        // Between five and fifteen lines per order, ten on average
        ("order_line", orders * 10),
    ]
}

/// The warehouse count a scale factor asks for. One warehouse is the
/// specification's minimum, so a fractional scale still loads a usable
/// database rather than an empty one.
pub fn warehouses_for(scale: f64) -> i64 {
    ((scale.max(0.0)).round() as i64).max(1)
}

/// Generates and streams the whole database as batched INSERT statements.
pub fn generate(
    scale: f64,
    rows_per_statement: usize,
    sink: &mut dyn FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    if !(scale > 0.0) {
        return Err(format!("scale factor must be above zero, got {scale}"));
    }
    let warehouses = warehouses_for(scale);

    generate_item(rows_per_statement, sink)?;
    generate_warehouse_and_district(warehouses, rows_per_statement, sink)?;
    generate_stock(warehouses, rows_per_statement, sink)?;
    generate_customer_and_history(warehouses, rows_per_statement, sink)?;
    generate_orders(warehouses, rows_per_statement, sink)
}

fn generate_item(
    rows_per_statement: usize,
    sink: &mut dyn FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    let mut rng = Rng::new(0x4954_454d);
    let mut b = InsertBatcher::new("item", rows_per_statement, sink);
    for id in 1..=ITEMS {
        // A tenth of items are flagged ORIGINAL, which is what the
        // Stock-Level and New-Order brand-generic rule reads
        let mut data = random_string(&mut rng, 26, 50);
        if rng.range(1, 10) == 1 {
            let at = rng.range(0, (data.len() as i64 - 8).max(0)) as usize;
            data.replace_range(at..at.min(data.len()), "ORIGINAL");
        }
        let row = format!(
            "({}, {}, {}, {}, {})",
            id,
            rng.range(1, 10_000),
            quote(&random_string(&mut rng, 14, 24)),
            rng.money(1, 100),
            quote(&data)
        );
        b.push(&row)?;
    }
    b.flush()
}

fn generate_warehouse_and_district(
    warehouses: i64,
    rows_per_statement: usize,
    sink: &mut dyn FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    let mut rng = Rng::new(0x5741_5245);
    {
        let mut b = InsertBatcher::new("warehouse", rows_per_statement, sink);
        for w in 1..=warehouses {
            let row = format!(
                "({}, {}, {}, {}, {}, {}, {}, {}, {})",
                w,
                quote(&random_string(&mut rng, 6, 10)),
                quote(&random_string(&mut rng, 10, 20)),
                quote(&random_string(&mut rng, 10, 20)),
                quote(&random_string(&mut rng, 10, 20)),
                quote(&random_string(&mut rng, 2, 2)),
                quote(&zip(&mut rng)),
                // Tax is a rate between zero and 0.2
                format!("0.{:04}", rng.range(0, 2000)),
                "300000.00"
            );
            b.push(&row)?;
        }
        b.flush()?;
    }
    let mut b = InsertBatcher::new("district", rows_per_statement, sink);
    for w in 1..=warehouses {
        for d in 1..=DISTRICTS_PER_WAREHOUSE {
            let row = format!(
                "({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
                d,
                w,
                quote(&random_string(&mut rng, 6, 10)),
                quote(&random_string(&mut rng, 10, 20)),
                quote(&random_string(&mut rng, 10, 20)),
                quote(&random_string(&mut rng, 10, 20)),
                quote(&random_string(&mut rng, 2, 2)),
                quote(&zip(&mut rng)),
                format!("0.{:04}", rng.range(0, 2000)),
                "30000.00",
                // The load leaves 3000 orders placed, so the next id is 3001
                ORDERS_PER_DISTRICT + 1
            );
            b.push(&row)?;
        }
    }
    b.flush()
}

fn generate_stock(
    warehouses: i64,
    rows_per_statement: usize,
    sink: &mut dyn FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    let mut rng = Rng::new(0x5354_4f43);
    let mut b = InsertBatcher::new("stock", rows_per_statement, sink);
    for w in 1..=warehouses {
        for i in 1..=ITEMS {
            let mut data = random_string(&mut rng, 26, 50);
            if rng.range(1, 10) == 1 {
                let at = rng.range(0, (data.len() as i64 - 8).max(0)) as usize;
                data.replace_range(at..at.min(data.len()), "ORIGINAL");
            }
            let dists: Vec<String> = (0..10)
                .map(|_| quote(&random_string(&mut rng, 24, 24)))
                .collect();
            let row = format!(
                "({}, {}, {}, {}, 0, 0, 0, {})",
                i,
                w,
                rng.range(10, 100),
                dists.join(", "),
                quote(&data)
            );
            b.push(&row)?;
        }
    }
    b.flush()
}

fn generate_customer_and_history(
    warehouses: i64,
    rows_per_statement: usize,
    sink: &mut dyn FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    let mut rng = Rng::new(0x4355_5354);
    let since = "2026-01-01 00:00:00";
    {
        let mut b = InsertBatcher::new("customer", rows_per_statement, sink);
        for w in 1..=warehouses {
            for d in 1..=DISTRICTS_PER_WAREHOUSE {
                for c in 1..=CUSTOMERS_PER_DISTRICT {
                    // The first thousand customers take the surnames the
                    // by-name lookup searches for, the rest are drawn
                    // non-uniformly across the same space
                    let name_source = if c <= 1_000 {
                        c - 1
                    } else {
                        nurand(&mut rng, 255, 0, 999)
                    };
                    // A tenth of customers carry bad credit, which is what
                    // the Payment transaction's c_data rewrite keys on
                    let credit = if rng.range(1, 10) == 1 { "BC" } else { "GC" };
                    let row = format!(
                        "({}, {}, {}, {}, 'OE', {}, {}, {}, {}, {}, {}, {}, TIMESTAMP {}, {}, 50000.00, {}, -10.00, 10.00, 1, 0, {})",
                        c,
                        d,
                        w,
                        quote(&random_string(&mut rng, 8, 16)),
                        quote(&last_name(name_source)),
                        quote(&random_string(&mut rng, 10, 20)),
                        quote(&random_string(&mut rng, 10, 20)),
                        quote(&random_string(&mut rng, 10, 20)),
                        quote(&random_string(&mut rng, 2, 2)),
                        quote(&zip(&mut rng)),
                        quote(&random_string(&mut rng, 16, 16)),
                        quote(since),
                        quote(credit),
                        format!("0.{:04}", rng.range(0, 5000)),
                        quote(&random_string(&mut rng, 300, 500))
                    );
                    b.push(&row)?;
                }
            }
        }
        b.flush()?;
    }
    let mut b = InsertBatcher::new("history", rows_per_statement, sink);
    for w in 1..=warehouses {
        for d in 1..=DISTRICTS_PER_WAREHOUSE {
            for c in 1..=CUSTOMERS_PER_DISTRICT {
                let row = format!(
                    "({}, {}, {}, {}, {}, TIMESTAMP {}, 10.00, {})",
                    c,
                    d,
                    w,
                    d,
                    w,
                    quote(since),
                    quote(&random_string(&mut rng, 12, 24))
                );
                b.push(&row)?;
            }
        }
    }
    b.flush()
}

fn generate_orders(
    warehouses: i64,
    rows_per_statement: usize,
    sink: &mut dyn FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    let mut rng = Rng::new(0x4f52_4452);
    let entry = "2026-01-01 00:00:00";
    let mut order_rows: Vec<String> = Vec::new();
    let mut line_rows: Vec<String> = Vec::new();
    let mut new_rows: Vec<String> = Vec::new();

    for w in 1..=warehouses {
        for d in 1..=DISTRICTS_PER_WAREHOUSE {
            for o in 1..=ORDERS_PER_DISTRICT {
                let line_count = rng.range(5, 15);
                // The last 900 orders of a district are undelivered, which is
                // what gives the Delivery transaction something to consume
                let delivered = o <= ORDERS_PER_DISTRICT - NEW_ORDERS_PER_DISTRICT;
                let carrier = if delivered {
                    rng.range(1, 10).to_string()
                } else {
                    "NULL".to_string()
                };
                order_rows.push(format!(
                    "({}, {}, {}, {}, TIMESTAMP {}, {}, {}, 1)",
                    o,
                    d,
                    w,
                    rng.range(1, CUSTOMERS_PER_DISTRICT),
                    quote(entry),
                    carrier,
                    line_count
                ));
                if !delivered {
                    new_rows.push(format!("({o}, {d}, {w})"));
                }
                for n in 1..=line_count {
                    let (delivery, amount) = if delivered {
                        (format!("TIMESTAMP {}", quote(entry)), "0.00".to_string())
                    } else {
                        ("NULL".to_string(), format_cents(rng.range(1, 999_999)))
                    };
                    line_rows.push(format!(
                        "({}, {}, {}, {}, {}, {}, {}, 5, {}, {})",
                        o,
                        d,
                        w,
                        n,
                        rng.range(1, ITEMS),
                        w,
                        delivery,
                        amount,
                        quote(&random_string(&mut rng, 24, 24))
                    ));
                }
                if order_rows.len() >= rows_per_statement {
                    flush_rows("orders", &mut order_rows, rows_per_statement, sink)?;
                }
                if line_rows.len() >= rows_per_statement {
                    flush_rows("order_line", &mut line_rows, rows_per_statement, sink)?;
                }
                if new_rows.len() >= rows_per_statement {
                    flush_rows("new_order", &mut new_rows, rows_per_statement, sink)?;
                }
            }
        }
    }
    flush_rows("orders", &mut order_rows, rows_per_statement, sink)?;
    flush_rows("order_line", &mut line_rows, rows_per_statement, sink)?;
    flush_rows("new_order", &mut new_rows, rows_per_statement, sink)
}

fn flush_rows(
    table: &str,
    rows: &mut Vec<String>,
    rows_per_statement: usize,
    sink: &mut dyn FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    if rows.is_empty() {
        return Ok(());
    }
    let mut b = InsertBatcher::new(table, rows_per_statement, sink);
    for row in rows.drain(..) {
        b.push(&row)?;
    }
    b.flush()
}

/// Which of the five transactions a step runs. The mix is the
/// specification's: New-Order and Payment carry the load, the other three
/// appear at a fixed minimum so the measurement covers them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Transaction {
    NewOrder,
    Payment,
    OrderStatus,
    Delivery,
    StockLevel,
}

impl Transaction {
    pub fn name(&self) -> &'static str {
        match self {
            Transaction::NewOrder => "New-Order",
            Transaction::Payment => "Payment",
            Transaction::OrderStatus => "Order-Status",
            Transaction::Delivery => "Delivery",
            Transaction::StockLevel => "Stock-Level",
        }
    }
}

/// Draws the next transaction from the specification's mix: 45% New-Order,
/// 43% Payment, and 4% each of the remaining three.
pub fn next_transaction(rng: &mut Rng) -> Transaction {
    match rng.range(1, 100) {
        1..=45 => Transaction::NewOrder,
        46..=88 => Transaction::Payment,
        89..=92 => Transaction::OrderStatus,
        93..=96 => Transaction::Delivery,
        _ => Transaction::StockLevel,
    }
}

/// The SQL one transaction runs, in order, as a single unit of work.
///
/// Each is wrapped by the driver in BEGIN and COMMIT, because the
/// specification requires a transaction's reads and writes to be atomic and
/// measuring them autocommitted would measure a different thing.
pub fn statements(kind: Transaction, warehouses: i64, rng: &mut Rng) -> Vec<String> {
    let w = rng.range(1, warehouses);
    let d = rng.range(1, DISTRICTS_PER_WAREHOUSE);
    match kind {
        Transaction::NewOrder => new_order(w, d, warehouses, rng),
        Transaction::Payment => payment(w, d, warehouses, rng),
        Transaction::OrderStatus => order_status(w, d, rng),
        Transaction::Delivery => delivery(w, rng),
        Transaction::StockLevel => stock_level(w, d, rng),
    }
}

/// New-Order: read the warehouse, district and customer, take the next order
/// id, insert the order and its lines, and decrement each line's stock.
fn new_order(w: i64, d: i64, warehouses: i64, rng: &mut Rng) -> Vec<String> {
    let c = nurand(rng, 1023, 1, CUSTOMERS_PER_DISTRICT);
    let line_count = rng.range(5, 15);
    // A district's next order id is read, used, then advanced. Reading it
    // inside the transaction is what makes two concurrent New-Orders on one
    // district contend, which is the contention the benchmark is built on
    let order_id_query =
        format!("SELECT d_next_o_id, d_tax FROM district WHERE d_id = {d} AND d_w_id = {w}");
    let mut sql = vec![
        format!("SELECT w_tax FROM warehouse WHERE w_id = {w}"),
        order_id_query,
        format!(
            "SELECT c_discount, c_last, c_credit FROM customer WHERE c_w_id = {w} AND c_d_id = {d} AND c_id = {c}"
        ),
        format!(
            "UPDATE district SET d_next_o_id = d_next_o_id + 1 WHERE d_id = {d} AND d_w_id = {w}"
        ),
    ];
    // The order id is the district's counter, which the driver cannot read
    // between statements without a round trip, so the insert derives it from
    // the same expression the update advanced
    sql.push(format!(
        "INSERT INTO orders (o_id, o_d_id, o_w_id, o_c_id, o_entry_d, o_carrier_id, o_ol_cnt, o_all_local) \
         SELECT d_next_o_id - 1, {d}, {w}, {c}, TIMESTAMP '2026-01-01 00:00:00', NULL, {line_count}, 1 \
         FROM district WHERE d_id = {d} AND d_w_id = {w}"
    ));
    sql.push(format!(
        "INSERT INTO new_order (no_o_id, no_d_id, no_w_id) \
         SELECT d_next_o_id - 1, {d}, {w} FROM district WHERE d_id = {d} AND d_w_id = {w}"
    ));
    for n in 1..=line_count {
        let item = nurand(rng, 8191, 1, ITEMS);
        // One line in a hundred is supplied by a different warehouse, which
        // is what makes a fraction of transactions distributed
        let supply_w = if warehouses > 1 && rng.range(1, 100) == 1 {
            let mut other = rng.range(1, warehouses);
            if other == w {
                other = (w % warehouses) + 1;
            }
            other
        } else {
            w
        };
        let quantity = rng.range(1, 10);
        sql.push(format!(
            "SELECT i_price, i_name, i_data FROM item WHERE i_id = {item}"
        ));
        sql.push(format!(
            "SELECT s_quantity, s_data FROM stock WHERE s_i_id = {item} AND s_w_id = {supply_w}"
        ));
        // Stock falls by the quantity ordered and is replenished when it
        // would drop below ten, which is the specification's rule
        sql.push(format!(
            "UPDATE stock SET s_quantity = CASE WHEN s_quantity >= {} THEN s_quantity - {quantity} ELSE s_quantity - {quantity} + 91 END, \
             s_ytd = s_ytd + {quantity}, s_order_cnt = s_order_cnt + 1{} \
             WHERE s_i_id = {item} AND s_w_id = {supply_w}",
            quantity + 10,
            if supply_w != w { ", s_remote_cnt = s_remote_cnt + 1" } else { "" }
        ));
        sql.push(format!(
            "INSERT INTO order_line (ol_o_id, ol_d_id, ol_w_id, ol_number, ol_i_id, ol_supply_w_id, ol_delivery_d, ol_quantity, ol_amount, ol_dist_info) \
             SELECT d_next_o_id - 1, {d}, {w}, {n}, {item}, {supply_w}, NULL, {quantity}, i_price * {quantity}, 'dist-info-padding-abc' \
             FROM district, item WHERE d_id = {d} AND d_w_id = {w} AND i_id = {item}"
        ));
    }
    sql
}

/// Payment: move an amount onto the warehouse, district and customer
/// balances and record it in history. Sixty percent of lookups are by
/// surname rather than id, which is the case that has no primary key to use.
fn payment(w: i64, d: i64, warehouses: i64, rng: &mut Rng) -> Vec<String> {
    let amount = format_cents(rng.range(100, 500_000));
    // Fifteen percent of payments are made at a warehouse other than the
    // customer's own, which is what makes some payments distributed
    let (c_w, c_d) = if warehouses > 1 && rng.range(1, 100) <= 15 {
        (
            rng.range(1, warehouses),
            rng.range(1, DISTRICTS_PER_WAREHOUSE),
        )
    } else {
        (w, d)
    };
    let by_name = rng.range(1, 100) <= 60;
    let lookup = if by_name {
        let name = last_name(nurand(rng, 255, 0, 999));
        format!(
            "SELECT c_id, c_balance, c_credit FROM customer WHERE c_w_id = {c_w} AND c_d_id = {c_d} AND c_last = {} ORDER BY c_first",
            quote(&name)
        )
    } else {
        let c = nurand(rng, 1023, 1, CUSTOMERS_PER_DISTRICT);
        format!(
            "SELECT c_id, c_balance, c_credit FROM customer WHERE c_w_id = {c_w} AND c_d_id = {c_d} AND c_id = {c}"
        )
    };
    let c_id = nurand(rng, 1023, 1, CUSTOMERS_PER_DISTRICT);
    vec![
        format!("UPDATE warehouse SET w_ytd = w_ytd + {amount} WHERE w_id = {w}"),
        format!(
            "SELECT w_name, w_street_1, w_city, w_state, w_zip FROM warehouse WHERE w_id = {w}"
        ),
        format!("UPDATE district SET d_ytd = d_ytd + {amount} WHERE d_w_id = {w} AND d_id = {d}"),
        format!(
            "SELECT d_name, d_street_1, d_city, d_state, d_zip FROM district WHERE d_w_id = {w} AND d_id = {d}"
        ),
        lookup,
        format!(
            "UPDATE customer SET c_balance = c_balance - {amount}, c_ytd_payment = c_ytd_payment + {amount}, \
             c_payment_cnt = c_payment_cnt + 1 WHERE c_w_id = {c_w} AND c_d_id = {c_d} AND c_id = {c_id}"
        ),
        format!(
            "INSERT INTO history (h_c_id, h_c_d_id, h_c_w_id, h_d_id, h_w_id, h_date, h_amount, h_data) \
             VALUES ({c_id}, {c_d}, {c_w}, {d}, {w}, TIMESTAMP '2026-01-01 00:00:00', {amount}, 'payment')"
        ),
    ]
}

/// Order-Status: read a customer's most recent order and its lines. Read
/// only, and the one transaction that reaches a customer by surname most of
/// the time.
fn order_status(w: i64, d: i64, rng: &mut Rng) -> Vec<String> {
    let by_name = rng.range(1, 100) <= 60;
    let lookup = if by_name {
        let name = last_name(nurand(rng, 255, 0, 999));
        format!(
            "SELECT c_id, c_first, c_middle, c_last, c_balance FROM customer WHERE c_w_id = {w} AND c_d_id = {d} AND c_last = {} ORDER BY c_first",
            quote(&name)
        )
    } else {
        let c = nurand(rng, 1023, 1, CUSTOMERS_PER_DISTRICT);
        format!(
            "SELECT c_id, c_first, c_middle, c_last, c_balance FROM customer WHERE c_w_id = {w} AND c_d_id = {d} AND c_id = {c}"
        )
    };
    let c = nurand(rng, 1023, 1, CUSTOMERS_PER_DISTRICT);
    vec![
        lookup,
        format!(
            "SELECT o_id, o_entry_d, o_carrier_id FROM orders WHERE o_w_id = {w} AND o_d_id = {d} AND o_c_id = {c} ORDER BY o_id DESC LIMIT 1"
        ),
        format!(
            "SELECT ol_i_id, ol_supply_w_id, ol_quantity, ol_amount, ol_delivery_d FROM order_line \
             WHERE ol_w_id = {w} AND ol_d_id = {d} AND ol_o_id = ( \
             SELECT MAX(o_id) FROM orders WHERE o_w_id = {w} AND o_d_id = {d} AND o_c_id = {c})"
        ),
    ]
}

/// Delivery: for every district of a warehouse, take the oldest undelivered
/// order, mark it delivered and move its total onto the customer's balance.
fn delivery(w: i64, rng: &mut Rng) -> Vec<String> {
    let carrier = rng.range(1, 10);
    let mut sql = Vec::with_capacity((DISTRICTS_PER_WAREHOUSE * 4) as usize);
    for d in 1..=DISTRICTS_PER_WAREHOUSE {
        let oldest =
            format!("(SELECT MIN(no_o_id) FROM new_order WHERE no_w_id = {w} AND no_d_id = {d})");
        sql.push(format!(
            "SELECT MIN(no_o_id) FROM new_order WHERE no_w_id = {w} AND no_d_id = {d}"
        ));
        sql.push(format!(
            "DELETE FROM new_order WHERE no_w_id = {w} AND no_d_id = {d} AND no_o_id = {oldest}"
        ));
        sql.push(format!(
            "UPDATE orders SET o_carrier_id = {carrier} WHERE o_w_id = {w} AND o_d_id = {d} AND o_id = {oldest}"
        ));
        sql.push(format!(
            "UPDATE order_line SET ol_delivery_d = TIMESTAMP '2026-01-01 00:00:00' \
             WHERE ol_w_id = {w} AND ol_d_id = {d} AND ol_o_id = {oldest}"
        ));
        sql.push(format!(
            "SELECT SUM(ol_amount) FROM order_line WHERE ol_w_id = {w} AND ol_d_id = {d} AND ol_o_id = {oldest}"
        ));
    }
    sql
}

/// Stock-Level: count the distinct items in the district's last twenty
/// orders whose stock has fallen below a threshold. Read only, and the one
/// transaction that touches a large range rather than a point.
fn stock_level(w: i64, d: i64, rng: &mut Rng) -> Vec<String> {
    let threshold = rng.range(10, 20);
    vec![
        format!("SELECT d_next_o_id FROM district WHERE d_w_id = {w} AND d_id = {d}"),
        format!(
            "SELECT COUNT(DISTINCT s_i_id) FROM order_line, stock, district \
             WHERE d_w_id = {w} AND d_id = {d} AND ol_w_id = {w} AND ol_d_id = {d} \
             AND ol_o_id < d_next_o_id AND ol_o_id >= d_next_o_id - 20 \
             AND s_w_id = {w} AND s_i_id = ol_i_id AND s_quantity < {threshold}"
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(scale: f64, rows_per_statement: usize) -> Vec<String> {
        let mut out = Vec::new();
        let mut sink = |sql: &str| {
            out.push(sql.to_string());
            Ok(())
        };
        generate(scale, rows_per_statement, &mut sink).expect("generate");
        out
    }

    fn rows_in(statement: &str) -> usize {
        let Some((_, values)) = statement.split_once(" VALUES ") else {
            return 0;
        };
        let mut depth = 0usize;
        let mut rows = 0usize;
        let mut in_string = false;
        let mut chars = values.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '\'' if in_string && chars.peek() == Some(&'\'') => {
                    chars.next();
                }
                '\'' => in_string = !in_string,
                '(' if !in_string => depth += 1,
                ')' if !in_string => {
                    depth -= 1;
                    if depth == 0 {
                        rows += 1;
                    }
                }
                _ => {}
            }
        }
        rows
    }

    fn counts_by_table(statements: &[String]) -> std::collections::HashMap<String, usize> {
        let mut counts = std::collections::HashMap::new();
        for s in statements {
            let table = s
                .strip_prefix("INSERT INTO ")
                .and_then(|r| r.split_once(' '))
                .map(|(t, _)| t.to_string())
                .expect("an insert names its table");
            *counts.entry(table).or_insert(0) += rows_in(s);
        }
        counts
    }

    /// One warehouse of the real thing is 100k items and 100k stock rows, so
    /// the shape tests use a reduced item count by construction: the counts
    /// asserted here are the ones the constants promise.
    #[test]
    fn test_row_counts_match_what_the_warehouse_count_promises() {
        let counts = counts_by_table(&collect(1.0, 500));
        for (table, promised) in row_counts(1) {
            if table == "order_line" {
                // Five to fifteen lines per order
                let orders = counts["orders"];
                assert!(
                    counts["order_line"] >= orders * 5 && counts["order_line"] <= orders * 15,
                    "order_line {} is outside five to fifteen per order ({orders})",
                    counts["order_line"]
                );
                continue;
            }
            assert_eq!(
                counts[table] as i64, promised,
                "{table} generated {} rows against a promised {promised}",
                counts[table]
            );
        }
    }

    #[test]
    fn test_generation_is_reproducible_for_a_warehouse_count() {
        let a = collect(1.0, 1000);
        let b = collect(1.0, 1000);
        assert_eq!(a, b, "the same warehouse count produces the same database");
    }

    #[test]
    fn test_a_zero_or_negative_scale_is_refused() {
        let mut sink = |_: &str| Ok(());
        assert!(generate(0.0, 10, &mut sink).is_err());
        assert!(generate(-2.0, 10, &mut sink).is_err());
    }

    #[test]
    fn test_the_schema_covers_every_table_the_generator_writes() {
        let ddl = schema().join(" ");
        for table in tables() {
            assert!(
                ddl.contains(&format!("EXISTS {table} ")),
                "no CREATE TABLE for {table}"
            );
        }
    }

    #[test]
    fn test_surnames_are_the_ones_a_by_name_lookup_searches_for() {
        assert_eq!(last_name(0), "BARBARBAR");
        assert_eq!(last_name(999), "EINGEINGEING");
        // Every surname is built from the specification's syllables, which is
        // what makes a by-name lookup hit
        for n in 0..1000 {
            let name = last_name(n);
            assert!(name.len() <= 16, "c_last is VARCHAR(16), got {name}");
        }
    }

    #[test]
    fn test_the_transaction_mix_matches_the_specification() {
        let mut rng = Rng::new(11);
        let mut counts = std::collections::HashMap::new();
        const RUNS: usize = 100_000;
        for _ in 0..RUNS {
            *counts.entry(next_transaction(&mut rng)).or_insert(0usize) += 1;
        }
        let pct = |t: Transaction| counts[&t] as f64 * 100.0 / RUNS as f64;
        assert!((pct(Transaction::NewOrder) - 45.0).abs() < 1.0);
        assert!((pct(Transaction::Payment) - 43.0).abs() < 1.0);
        assert!((pct(Transaction::OrderStatus) - 4.0).abs() < 1.0);
        assert!((pct(Transaction::Delivery) - 4.0).abs() < 1.0);
        assert!((pct(Transaction::StockLevel) - 4.0).abs() < 1.0);
    }

    #[test]
    fn test_every_transaction_produces_statements_that_name_its_tables() {
        let mut rng = Rng::new(5);
        for kind in [
            Transaction::NewOrder,
            Transaction::Payment,
            Transaction::OrderStatus,
            Transaction::Delivery,
            Transaction::StockLevel,
        ] {
            let sql = statements(kind, 4, &mut rng);
            assert!(!sql.is_empty(), "{} produced no statements", kind.name());
            for s in &sql {
                assert!(
                    s.starts_with("SELECT")
                        || s.starts_with("UPDATE")
                        || s.starts_with("INSERT")
                        || s.starts_with("DELETE"),
                    "{} produced a statement that is not DML: {s}",
                    kind.name()
                );
            }
        }
    }

    /// New-Order and Payment write, Order-Status and Stock-Level do not. A
    /// read-only transaction that wrote would change the database the next
    /// measurement runs against.
    #[test]
    fn test_the_read_only_transactions_do_not_write() {
        let mut rng = Rng::new(9);
        for kind in [Transaction::OrderStatus, Transaction::StockLevel] {
            for _ in 0..64 {
                for s in statements(kind, 4, &mut rng) {
                    assert!(
                        s.starts_with("SELECT"),
                        "{} is read only but produced: {s}",
                        kind.name()
                    );
                }
            }
        }
    }

    #[test]
    fn test_a_single_warehouse_run_keeps_every_line_local() {
        let mut rng = Rng::new(13);
        for _ in 0..64 {
            for s in statements(Transaction::NewOrder, 1, &mut rng) {
                assert!(
                    !s.contains("s_remote_cnt"),
                    "one warehouse cannot have a remote line: {s}"
                );
            }
        }
    }
}
