//! TPC-H schema, data generation and the 22 query set.
//!
//! Cardinalities, value ranges and text vocabularies follow the TPC-H
//! specification, because the queries are written against them: `Q9` matches
//! `p_name like '%green%'`, `Q13` counts orders whose comment does not match
//! `%special%requests%`, `Q14` splits on `p_type like 'PROMO%'`. A generator
//! that invented its own text would leave those predicates matching nothing
//! and the numbers would describe an empty result rather than the workload.

use crate::{InsertBatcher, Rng, civil_from_days, days_from_civil, format_cents, quote};

/// Base row counts at scale factor 1. Every table except the two fixed ones
/// scales linearly, which is what makes a scale factor mean what it says.
const SUPPLIERS_PER_SF: i64 = 10_000;
const CUSTOMERS_PER_SF: i64 = 150_000;
const PARTS_PER_SF: i64 = 200_000;
const ORDERS_PER_SF: i64 = 1_500_000;
/// Every part has exactly four suppliers
const SUPPLIERS_PER_PART: i64 = 4;

/// The order date window the specification defines, and the window every
/// dated predicate in the query set is written against
const DATE_LOW: (i64, u32, u32) = (1992, 1, 1);
const DATE_HIGH: (i64, u32, u32) = (1998, 8, 2);

const REGIONS: [&str; 5] = ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"];

/// Nation name and the region it belongs to, in nationkey order
const NATIONS: [(&str, i64); 25] = [
    ("ALGERIA", 0),
    ("ARGENTINA", 1),
    ("BRAZIL", 1),
    ("CANADA", 1),
    ("EGYPT", 4),
    ("ETHIOPIA", 0),
    ("FRANCE", 3),
    ("GERMANY", 3),
    ("INDIA", 2),
    ("INDONESIA", 2),
    ("IRAN", 4),
    ("IRAQ", 4),
    ("JAPAN", 2),
    ("JORDAN", 4),
    ("KENYA", 0),
    ("MOROCCO", 0),
    ("MOZAMBIQUE", 0),
    ("PERU", 1),
    ("CHINA", 2),
    ("ROMANIA", 3),
    ("SAUDI ARABIA", 4),
    ("VIETNAM", 2),
    ("RUSSIA", 3),
    ("UNITED KINGDOM", 3),
    ("UNITED STATES", 1),
];

/// The 92 colours the specification draws `p_name` from. `Q9` selects on
/// `green`, so the list is the one the query expects.
const COLORS: [&str; 92] = [
    "almond",
    "antique",
    "aquamarine",
    "azure",
    "beige",
    "bisque",
    "black",
    "blanched",
    "blue",
    "blush",
    "brown",
    "burlywood",
    "burnished",
    "chartreuse",
    "chiffon",
    "chocolate",
    "coral",
    "cornflower",
    "cornsilk",
    "cream",
    "cyan",
    "dark",
    "deep",
    "dim",
    "dodger",
    "drab",
    "firebrick",
    "floral",
    "forest",
    "frosted",
    "gainsboro",
    "ghost",
    "goldenrod",
    "green",
    "grey",
    "honeydew",
    "hot",
    "indian",
    "ivory",
    "khaki",
    "lace",
    "lavender",
    "lawn",
    "lemon",
    "light",
    "lime",
    "linen",
    "magenta",
    "maroon",
    "medium",
    "metallic",
    "midnight",
    "mint",
    "misty",
    "moccasin",
    "navajo",
    "navy",
    "olive",
    "orange",
    "orchid",
    "pale",
    "papaya",
    "peach",
    "peru",
    "pink",
    "plum",
    "powder",
    "puff",
    "purple",
    "red",
    "rose",
    "rosy",
    "royal",
    "saddle",
    "salmon",
    "sandy",
    "seashell",
    "sienna",
    "sky",
    "slate",
    "smoke",
    "snow",
    "spring",
    "steel",
    "tan",
    "thistle",
    "tomato",
    "turquoise",
    "violet",
    "wheat",
    "white",
    "yellow",
];

const TYPE_SYLLABLE_1: [&str; 6] = ["STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO"];
const TYPE_SYLLABLE_2: [&str; 5] = ["ANODIZED", "BURNISHED", "PLATED", "POLISHED", "BRUSHED"];
const TYPE_SYLLABLE_3: [&str; 5] = ["TIN", "NICKEL", "BRASS", "STEEL", "COPPER"];

const CONTAINER_SYLLABLE_1: [&str; 5] = ["SM", "LG", "MED", "JUMBO", "WRAP"];
const CONTAINER_SYLLABLE_2: [&str; 8] = ["CASE", "BOX", "BAG", "JAR", "PKG", "PACK", "CAN", "DRUM"];

const SEGMENTS: [&str; 5] = [
    "AUTOMOBILE",
    "BUILDING",
    "FURNITURE",
    "MACHINERY",
    "HOUSEHOLD",
];

const PRIORITIES: [&str; 5] = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"];

const INSTRUCTIONS: [&str; 4] = [
    "DELIVER IN PERSON",
    "COLLECT COD",
    "NONE",
    "TAKE BACK RETURN",
];

const MODES: [&str; 7] = ["REG AIR", "AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB"];

// The comment grammar's vocabulary. Kept small enough to read and wide
// enough that the LIKE predicates in the query set match a meaningful
// fraction rather than everything or nothing.
const NOUNS: [&str; 24] = [
    "foxes",
    "ideas",
    "theodolites",
    "pinto beans",
    "instructions",
    "dependencies",
    "excuses",
    "platelets",
    "asymptotes",
    "courts",
    "dolphins",
    "multipliers",
    "sauternes",
    "warthogs",
    "frets",
    "dinos",
    "attainments",
    "somas",
    "Tiresias",
    "patterns",
    "forges",
    "braids",
    "hockey players",
    "frays",
];
const VERBS: [&str; 20] = [
    "sleep",
    "wake",
    "are",
    "cajole",
    "haggle",
    "nag",
    "use",
    "boost",
    "affix",
    "detect",
    "integrate",
    "maintain",
    "nod",
    "was",
    "lose",
    "sublate",
    "solve",
    "thrash",
    "promise",
    "engage",
];
const ADJECTIVES: [&str; 20] = [
    "furious",
    "sly",
    "careful",
    "blithe",
    "quick",
    "fluffy",
    "slow",
    "quiet",
    "ruthless",
    "thin",
    "close",
    "dogged",
    "daring",
    "brave",
    "stealthy",
    "permanent",
    "enticing",
    "idle",
    "busy",
    "regular",
];
const ADVERBS: [&str; 16] = [
    "sometimes",
    "always",
    "never",
    "furiously",
    "slyly",
    "carefully",
    "blithely",
    "quickly",
    "fluffily",
    "slowly",
    "quietly",
    "ruthlessly",
    "thinly",
    "closely",
    "doggedly",
    "daringly",
];
const PREPOSITIONS: [&str; 16] = [
    "about",
    "above",
    "according to",
    "across",
    "after",
    "against",
    "along",
    "alongside of",
    "among",
    "around",
    "at",
    "atop",
    "before",
    "between",
    "beyond",
    "by",
];
/// `Q13` counts the orders whose comment does not match `%special%requests%`,
/// so the grammar has to be able to produce that pair in that order.
const AUXILIARIES: [&str; 8] = [
    "special", "pending", "unusual", "express", "final", "bold", "even", "silent",
];
const TERMINATORS: [&str; 6] = [
    "requests",
    "accounts",
    "packages",
    "deposits",
    "asymptotes",
    "instructions",
];

/// Builds one comment of at most `max_len` characters from the grammar.
fn comment(rng: &mut Rng, max_len: usize) -> String {
    let mut text = String::with_capacity(max_len);
    // Enough clauses to fill a long column, trimmed to the declared width
    while text.len() < max_len {
        if !text.is_empty() {
            text.push(' ');
        }
        match rng.range(0, 4) {
            0 => text.push_str(&format!(
                "{} {} {} {}",
                rng.pick(&ADJECTIVES),
                rng.pick(&NOUNS),
                rng.pick(&VERBS),
                rng.pick(&ADVERBS)
            )),
            1 => text.push_str(&format!(
                "{} {} {} {}",
                rng.pick(&ADVERBS),
                rng.pick(&VERBS),
                rng.pick(&AUXILIARIES),
                rng.pick(&TERMINATORS)
            )),
            2 => text.push_str(&format!(
                "{} {} {} {} {}",
                rng.pick(&NOUNS),
                rng.pick(&VERBS),
                rng.pick(&PREPOSITIONS),
                rng.pick(&AUXILIARIES),
                rng.pick(&TERMINATORS)
            )),
            3 => text.push_str(&format!(
                "{} {} {}",
                rng.pick(&AUXILIARIES),
                rng.pick(&TERMINATORS),
                rng.pick(&VERBS)
            )),
            _ => text.push_str(&format!(
                "{} {} {}",
                rng.pick(&ADJECTIVES),
                rng.pick(&AUXILIARIES),
                rng.pick(&TERMINATORS)
            )),
        }
    }
    // Cut on a character boundary, the vocabulary is ASCII so a byte index
    // is one, and trim a partial trailing word
    text.truncate(max_len);
    while !text.is_empty() && !text.ends_with(' ') && text.len() == max_len {
        text.pop();
    }
    text.trim_end().to_string()
}

/// A phone number in the specification's `CC-AAA-BBB-CCCC` shape, whose
/// country code is the nation key plus ten. `Q22` selects on the first two
/// characters, so the encoding has to be the one it expects.
fn phone(rng: &mut Rng, nation_key: i64) -> String {
    format!(
        "{:02}-{:03}-{:03}-{:04}",
        nation_key + 10,
        rng.range(100, 999),
        rng.range(100, 999),
        rng.range(1000, 9999)
    )
}

/// The DDL for the eight TPC-H tables, in dependency order.
pub fn schema() -> Vec<&'static str> {
    vec![
        "CREATE TABLE IF NOT EXISTS region (r_regionkey INT, r_name VARCHAR(25), r_comment VARCHAR(152))",
        "CREATE TABLE IF NOT EXISTS nation (n_nationkey INT, n_name VARCHAR(25), n_regionkey INT, n_comment VARCHAR(152))",
        "CREATE TABLE IF NOT EXISTS part (p_partkey INT, p_name VARCHAR(55), p_mfgr VARCHAR(25), p_brand VARCHAR(10), p_type VARCHAR(25), p_size INT, p_container VARCHAR(10), p_retailprice DECIMAL(15,2), p_comment VARCHAR(23))",
        "CREATE TABLE IF NOT EXISTS supplier (s_suppkey INT, s_name VARCHAR(25), s_address VARCHAR(40), s_nationkey INT, s_phone VARCHAR(15), s_acctbal DECIMAL(15,2), s_comment VARCHAR(101))",
        "CREATE TABLE IF NOT EXISTS partsupp (ps_partkey INT, ps_suppkey INT, ps_availqty INT, ps_supplycost DECIMAL(15,2), ps_comment VARCHAR(199))",
        "CREATE TABLE IF NOT EXISTS customer (c_custkey INT, c_name VARCHAR(25), c_address VARCHAR(40), c_nationkey INT, c_phone VARCHAR(15), c_acctbal DECIMAL(15,2), c_mktsegment VARCHAR(10), c_comment VARCHAR(117))",
        "CREATE TABLE IF NOT EXISTS orders (o_orderkey INT, o_custkey INT, o_orderstatus VARCHAR(1), o_totalprice DECIMAL(15,2), o_orderdate DATE, o_orderpriority VARCHAR(15), o_clerk VARCHAR(15), o_shippriority INT, o_comment VARCHAR(79))",
        "CREATE TABLE IF NOT EXISTS lineitem (l_orderkey INT, l_partkey INT, l_suppkey INT, l_linenumber INT, l_quantity DECIMAL(15,2), l_extendedprice DECIMAL(15,2), l_discount DECIMAL(15,2), l_tax DECIMAL(15,2), l_returnflag VARCHAR(1), l_linestatus VARCHAR(1), l_shipdate DATE, l_commitdate DATE, l_receiptdate DATE, l_shipinstruct VARCHAR(25), l_shipmode VARCHAR(10), l_comment VARCHAR(44))",
    ]
}

/// The tables the generator writes, so a driver can clear a previous run.
pub fn tables() -> [&'static str; 8] {
    [
        "lineitem", "orders", "partsupp", "part", "supplier", "customer", "nation", "region",
    ]
}

/// Row counts this scale factor produces, in `tables()` order. Reported
/// before a load so an operator sees the size they asked for.
pub fn row_counts(scale: f64) -> Vec<(&'static str, i64)> {
    let s = |base: i64| ((base as f64) * scale).round() as i64;
    let parts = s(PARTS_PER_SF);
    let orders = s(ORDERS_PER_SF);
    vec![
        ("region", REGIONS.len() as i64),
        ("nation", NATIONS.len() as i64),
        ("part", parts),
        ("supplier", s(SUPPLIERS_PER_SF)),
        ("partsupp", parts * SUPPLIERS_PER_PART),
        ("customer", s(CUSTOMERS_PER_SF)),
        ("orders", orders),
        // Between one and seven lines per order, so four on average
        ("lineitem", orders * 4),
    ]
}

/// Generates and streams the whole database as batched INSERT statements.
///
/// Nothing is buffered beyond one statement, so a scale factor is limited by
/// what the server can store rather than by this process's memory.
pub fn generate(
    scale: f64,
    rows_per_statement: usize,
    sink: &mut dyn FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    if !(scale > 0.0) {
        return Err(format!("scale factor must be above zero, got {scale}"));
    }
    let s = |base: i64| (((base as f64) * scale).round() as i64).max(1);
    let suppliers = s(SUPPLIERS_PER_SF);
    let customers = s(CUSTOMERS_PER_SF);
    let parts = s(PARTS_PER_SF);
    let orders = s(ORDERS_PER_SF);

    generate_region(rows_per_statement, sink)?;
    generate_nation(rows_per_statement, sink)?;
    generate_part(parts, suppliers, rows_per_statement, sink)?;
    generate_supplier(suppliers, rows_per_statement, sink)?;
    generate_customer(customers, rows_per_statement, sink)?;
    generate_orders_and_lineitem(
        orders,
        customers,
        parts,
        suppliers,
        rows_per_statement,
        sink,
    )
}

fn generate_region(
    rows_per_statement: usize,
    sink: &mut dyn FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    let mut rng = Rng::new(0x5245_4749_4f4e);
    let mut b = InsertBatcher::new("region", rows_per_statement, sink);
    for (key, name) in REGIONS.iter().enumerate() {
        let row = format!(
            "({}, {}, {})",
            key,
            quote(name),
            quote(&comment(&mut rng, 152))
        );
        b.push(&row)?;
    }
    b.flush()
}

fn generate_nation(
    rows_per_statement: usize,
    sink: &mut dyn FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    let mut rng = Rng::new(0x4e41_5449_4f4e);
    let mut b = InsertBatcher::new("nation", rows_per_statement, sink);
    for (key, (name, region)) in NATIONS.iter().enumerate() {
        let row = format!(
            "({}, {}, {}, {})",
            key,
            quote(name),
            region,
            quote(&comment(&mut rng, 152))
        );
        b.push(&row)?;
    }
    b.flush()
}

fn generate_part(
    parts: i64,
    suppliers: i64,
    rows_per_statement: usize,
    sink: &mut dyn FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    let mut rng = Rng::new(0x5041_5254);
    let mut part_batch = InsertBatcher::new("part", rows_per_statement, sink);
    let mut part_rows: Vec<String> = Vec::with_capacity(parts as usize);
    for key in 1..=parts {
        // Five distinct colours, which is what makes `p_name like '%green%'`
        // select a stable fraction
        let mut chosen: Vec<&str> = Vec::with_capacity(5);
        while chosen.len() < 5 {
            let c = *rng.pick(&COLORS);
            if !chosen.contains(&c) {
                chosen.push(c);
            }
        }
        let name = chosen.join(" ");
        let manufacturer = rng.range(1, 5);
        let brand = manufacturer * 10 + rng.range(1, 5);
        let p_type = format!(
            "{} {} {}",
            rng.pick(&TYPE_SYLLABLE_1),
            rng.pick(&TYPE_SYLLABLE_2),
            rng.pick(&TYPE_SYLLABLE_3)
        );
        let container = format!(
            "{} {}",
            rng.pick(&CONTAINER_SYLLABLE_1),
            rng.pick(&CONTAINER_SYLLABLE_2)
        );
        // The specification's retail price formula, so prices spread across
        // the range the query set's predicates assume
        let price_cents = 90_000 + ((key / 10) % 20_001) + 100 * (key % 1_000);
        part_rows.push(format!(
            "({}, {}, {}, {}, {}, {}, {}, {}, {})",
            key,
            quote(&name),
            quote(&format!("Manufacturer#{manufacturer}")),
            quote(&format!("Brand#{brand}")),
            quote(&p_type),
            rng.range(1, 50),
            quote(&container),
            format_cents(price_cents),
            quote(&comment(&mut rng, 23))
        ));
        if part_rows.len() >= rows_per_statement {
            for row in part_rows.drain(..) {
                part_batch.push(&row)?;
            }
        }
    }
    for row in part_rows.drain(..) {
        part_batch.push(&row)?;
    }
    part_batch.flush()?;
    drop(part_batch);

    // Four suppliers per part, spread so no supplier carries every part
    let mut ps_rng = Rng::new(0x5053_5550);
    let mut ps_batch = InsertBatcher::new("partsupp", rows_per_statement, sink);
    for key in 1..=parts {
        for i in 0..SUPPLIERS_PER_PART {
            let supp = ((key + i * ((suppliers / SUPPLIERS_PER_PART) + 1)) % suppliers) + 1;
            let row = format!(
                "({}, {}, {}, {}, {})",
                key,
                supp,
                ps_rng.range(1, 9_999),
                ps_rng.money(1, 1_000),
                quote(&comment(&mut ps_rng, 199))
            );
            ps_batch.push(&row)?;
        }
    }
    ps_batch.flush()
}

fn generate_supplier(
    suppliers: i64,
    rows_per_statement: usize,
    sink: &mut dyn FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    let mut rng = Rng::new(0x5355_5050);
    let mut b = InsertBatcher::new("supplier", rows_per_statement, sink);
    for key in 1..=suppliers {
        let nation = rng.range(0, NATIONS.len() as i64 - 1);
        let row = format!(
            "({}, {}, {}, {}, {}, {}, {})",
            key,
            quote(&format!("Supplier#{key:09}")),
            quote(&comment(&mut rng, 40)),
            nation,
            quote(&phone(&mut rng, nation)),
            rng.money(-999, 9_999),
            quote(&comment(&mut rng, 101))
        );
        b.push(&row)?;
    }
    b.flush()
}

fn generate_customer(
    customers: i64,
    rows_per_statement: usize,
    sink: &mut dyn FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    let mut rng = Rng::new(0x4355_5354);
    let mut b = InsertBatcher::new("customer", rows_per_statement, sink);
    for key in 1..=customers {
        let nation = rng.range(0, NATIONS.len() as i64 - 1);
        let row = format!(
            "({}, {}, {}, {}, {}, {}, {}, {})",
            key,
            quote(&format!("Customer#{key:09}")),
            quote(&comment(&mut rng, 40)),
            nation,
            quote(&phone(&mut rng, nation)),
            rng.money(-999, 9_999),
            quote(rng.pick(&SEGMENTS)),
            quote(&comment(&mut rng, 117))
        );
        b.push(&row)?;
    }
    b.flush()
}

/// Orders and their lines together, because a line's dates and its order's
/// status are derived from the same order.
fn generate_orders_and_lineitem(
    orders: i64,
    customers: i64,
    parts: i64,
    suppliers: i64,
    rows_per_statement: usize,
    sink: &mut dyn FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    let low = days_from_civil(DATE_LOW.0, DATE_LOW.1, DATE_LOW.2);
    let high = days_from_civil(DATE_HIGH.0, DATE_HIGH.1, DATE_HIGH.2);

    let mut o_rng = Rng::new(0x4f52_4452);
    let mut l_rng = Rng::new(0x4c49_4e45);

    // Orders and lines are emitted per order so the generator holds one
    // order's lines at a time rather than the table
    let mut order_rows: Vec<String> = Vec::with_capacity(rows_per_statement);
    let mut line_rows: Vec<String> = Vec::with_capacity(rows_per_statement * 7);

    for key in 1..=orders {
        let order_date = o_rng.range(low, high);
        let line_count = l_rng.range(1, 7);

        // The order's status follows its lines: all received is F, none is O,
        // a mix is P. The lines are built first so the status is derived
        // rather than asserted
        let mut lines: Vec<(String, i64)> = Vec::with_capacity(line_count as usize);
        let mut total_cents: i64 = 0;
        for line_number in 1..=line_count {
            let part = l_rng.range(1, parts);
            let supp = l_rng.range(1, suppliers);
            let quantity = l_rng.range(1, 50);
            let retail_cents = 90_000 + ((part / 10) % 20_001) + 100 * (part % 1_000);
            let extended_cents = retail_cents * quantity;
            let discount_cents = l_rng.range(0, 10);
            let tax_cents = l_rng.range(0, 8);
            let ship_date = order_date + l_rng.range(1, 121);
            let commit_date = order_date + l_rng.range(30, 90);
            let receipt_date = ship_date + l_rng.range(1, 30);
            // A line is returned only once received, which is what makes the
            // returnflag distribution in Q1 meaningful
            let received = receipt_date <= high;
            let return_flag = if received {
                if l_rng.range(0, 1) == 0 { "R" } else { "A" }
            } else {
                "N"
            };
            let line_status = if ship_date <= high { "F" } else { "O" };
            total_cents += extended_cents - (extended_cents * discount_cents / 100)
                + (extended_cents * tax_cents / 100);
            lines.push((
                format!(
                    "({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, DATE {}, DATE {}, DATE {}, {}, {}, {})",
                    key,
                    part,
                    supp,
                    line_number,
                    format_cents(quantity * 100),
                    format_cents(extended_cents),
                    format_cents(discount_cents),
                    format_cents(tax_cents),
                    quote(return_flag),
                    quote(line_status),
                    quote(&civil_from_days(ship_date)),
                    quote(&civil_from_days(commit_date)),
                    quote(&civil_from_days(receipt_date)),
                    quote(l_rng.pick(&INSTRUCTIONS)),
                    quote(l_rng.pick(&MODES)),
                    quote(&comment(&mut l_rng, 44))
                ),
                if received { 1 } else { 0 },
            ));
        }
        let received_lines: i64 = lines.iter().map(|(_, r)| r).sum();
        let status = if received_lines == 0 {
            "O"
        } else if received_lines == line_count {
            "F"
        } else {
            "P"
        };

        order_rows.push(format!(
            "({}, {}, {}, {}, DATE {}, {}, {}, {}, {})",
            key,
            o_rng.range(1, customers),
            quote(status),
            format_cents(total_cents),
            quote(&civil_from_days(order_date)),
            quote(o_rng.pick(&PRIORITIES)),
            quote(&format!("Clerk#{:09}", o_rng.range(1, 1_000))),
            0,
            quote(&comment(&mut o_rng, 79))
        ));
        for (line, _) in lines {
            line_rows.push(line);
        }

        if order_rows.len() >= rows_per_statement {
            flush_rows("orders", &mut order_rows, rows_per_statement, sink)?;
        }
        if line_rows.len() >= rows_per_statement {
            flush_rows("lineitem", &mut line_rows, rows_per_statement, sink)?;
        }
    }
    flush_rows("orders", &mut order_rows, rows_per_statement, sink)?;
    flush_rows("lineitem", &mut line_rows, rows_per_statement, sink)
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

/// The 22 TPC-H queries, each with its number.
///
/// Substitution parameters are the specification's validation values, so two
/// runs compare and a result can be checked against a published answer.
pub fn queries() -> Vec<(&'static str, &'static str)> {
    vec![
        ("Q1", Q1),
        ("Q2", Q2),
        ("Q3", Q3),
        ("Q4", Q4),
        ("Q5", Q5),
        ("Q6", Q6),
        ("Q7", Q7),
        ("Q8", Q8),
        ("Q9", Q9),
        ("Q10", Q10),
        ("Q11", Q11),
        ("Q12", Q12),
        ("Q13", Q13),
        ("Q14", Q14),
        ("Q15", Q15),
        ("Q16", Q16),
        ("Q17", Q17),
        ("Q18", Q18),
        ("Q19", Q19),
        ("Q20", Q20),
        ("Q21", Q21),
        ("Q22", Q22),
    ]
}

const Q1: &str = "\
SELECT l_returnflag, l_linestatus, SUM(l_quantity) AS sum_qty, \
SUM(l_extendedprice) AS sum_base_price, \
SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price, \
SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge, \
AVG(l_quantity) AS avg_qty, AVG(l_extendedprice) AS avg_price, \
AVG(l_discount) AS avg_disc, COUNT(*) AS count_order \
FROM lineitem WHERE l_shipdate <= DATE '1998-09-02' \
GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus";

const Q2: &str = "\
SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment \
FROM part, supplier, partsupp, nation, region \
WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey AND p_size = 15 \
AND p_type LIKE '%BRASS' AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey \
AND r_name = 'EUROPE' AND ps_supplycost = ( \
SELECT MIN(ps_supplycost) FROM partsupp, supplier, nation, region \
WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey AND s_nationkey = n_nationkey \
AND n_regionkey = r_regionkey AND r_name = 'EUROPE') \
ORDER BY s_acctbal DESC, n_name, s_name, p_partkey LIMIT 100";

const Q3: &str = "\
SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue, o_orderdate, o_shippriority \
FROM customer, orders, lineitem \
WHERE c_mktsegment = 'BUILDING' AND c_custkey = o_custkey AND l_orderkey = o_orderkey \
AND o_orderdate < DATE '1995-03-15' AND l_shipdate > DATE '1995-03-15' \
GROUP BY l_orderkey, o_orderdate, o_shippriority \
ORDER BY revenue DESC, o_orderdate LIMIT 10";

const Q4: &str = "\
SELECT o_orderpriority, COUNT(*) AS order_count FROM orders \
WHERE o_orderdate >= DATE '1993-07-01' AND o_orderdate < DATE '1993-10-01' \
AND EXISTS (SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate) \
GROUP BY o_orderpriority ORDER BY o_orderpriority";

const Q5: &str = "\
SELECT n_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue \
FROM customer, orders, lineitem, supplier, nation, region \
WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey AND l_suppkey = s_suppkey \
AND c_nationkey = s_nationkey AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey \
AND r_name = 'ASIA' AND o_orderdate >= DATE '1994-01-01' AND o_orderdate < DATE '1995-01-01' \
GROUP BY n_name ORDER BY revenue DESC";

const Q6: &str = "\
SELECT SUM(l_extendedprice * l_discount) AS revenue FROM lineitem \
WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01' \
AND l_discount >= 0.05 AND l_discount <= 0.07 AND l_quantity < 24";

const Q7: &str = "\
SELECT supp_nation, cust_nation, l_year, SUM(volume) AS revenue FROM ( \
SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation, \
EXTRACT(YEAR FROM l_shipdate) AS l_year, l_extendedprice * (1 - l_discount) AS volume \
FROM supplier, lineitem, orders, customer, nation n1, nation n2 \
WHERE s_suppkey = l_suppkey AND o_orderkey = l_orderkey AND c_custkey = o_custkey \
AND s_nationkey = n1.n_nationkey AND c_nationkey = n2.n_nationkey \
AND ((n1.n_name = 'FRANCE' AND n2.n_name = 'GERMANY') \
OR (n1.n_name = 'GERMANY' AND n2.n_name = 'FRANCE')) \
AND l_shipdate >= DATE '1995-01-01' AND l_shipdate <= DATE '1996-12-31') AS shipping \
GROUP BY supp_nation, cust_nation, l_year ORDER BY supp_nation, cust_nation, l_year";

const Q8: &str = "\
SELECT o_year, SUM(CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0 END) / SUM(volume) AS mkt_share \
FROM (SELECT EXTRACT(YEAR FROM o_orderdate) AS o_year, \
l_extendedprice * (1 - l_discount) AS volume, n2.n_name AS nation \
FROM part, supplier, lineitem, orders, customer, nation n1, nation n2, region \
WHERE p_partkey = l_partkey AND s_suppkey = l_suppkey AND l_orderkey = o_orderkey \
AND o_custkey = c_custkey AND c_nationkey = n1.n_nationkey AND n1.n_regionkey = r_regionkey \
AND r_name = 'AMERICA' AND s_nationkey = n2.n_nationkey \
AND o_orderdate >= DATE '1995-01-01' AND o_orderdate <= DATE '1996-12-31' \
AND p_type = 'ECONOMY ANODIZED STEEL') AS all_nations \
GROUP BY o_year ORDER BY o_year";

const Q9: &str = "\
SELECT nation, o_year, SUM(amount) AS sum_profit FROM ( \
SELECT n_name AS nation, EXTRACT(YEAR FROM o_orderdate) AS o_year, \
l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount \
FROM part, supplier, lineitem, partsupp, orders, nation \
WHERE s_suppkey = l_suppkey AND ps_suppkey = l_suppkey AND ps_partkey = l_partkey \
AND p_partkey = l_partkey AND o_orderkey = l_orderkey AND s_nationkey = n_nationkey \
AND p_name LIKE '%green%') AS profit \
GROUP BY nation, o_year ORDER BY nation, o_year DESC";

const Q10: &str = "\
SELECT c_custkey, c_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue, c_acctbal, \
n_name, c_address, c_phone, c_comment \
FROM customer, orders, lineitem, nation \
WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey \
AND o_orderdate >= DATE '1993-10-01' AND o_orderdate < DATE '1994-01-01' \
AND l_returnflag = 'R' AND c_nationkey = n_nationkey \
GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment \
ORDER BY revenue DESC LIMIT 20";

const Q11: &str = "\
SELECT ps_partkey, SUM(ps_supplycost * ps_availqty) AS value \
FROM partsupp, supplier, nation \
WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY' \
GROUP BY ps_partkey HAVING SUM(ps_supplycost * ps_availqty) > ( \
SELECT SUM(ps_supplycost * ps_availqty) * 0.0001 FROM partsupp, supplier, nation \
WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY') \
ORDER BY value DESC";

const Q12: &str = "\
SELECT l_shipmode, \
SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count, \
SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count \
FROM orders, lineitem \
WHERE o_orderkey = l_orderkey AND l_shipmode IN ('MAIL', 'SHIP') \
AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate \
AND l_receiptdate >= DATE '1994-01-01' AND l_receiptdate < DATE '1995-01-01' \
GROUP BY l_shipmode ORDER BY l_shipmode";

const Q13: &str = "\
SELECT c_count, COUNT(*) AS custdist FROM ( \
SELECT c_custkey, COUNT(o_orderkey) AS c_count FROM customer LEFT OUTER JOIN orders \
ON c_custkey = o_custkey AND o_comment NOT LIKE '%special%requests%' \
GROUP BY c_custkey) AS c_orders \
GROUP BY c_count ORDER BY custdist DESC, c_count DESC";

const Q14: &str = "\
SELECT 100.00 * SUM(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END) \
/ SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue \
FROM lineitem, part WHERE l_partkey = p_partkey \
AND l_shipdate >= DATE '1995-09-01' AND l_shipdate < DATE '1995-10-01'";

const Q15: &str = "\
SELECT s_suppkey, s_name, s_address, s_phone, total_revenue FROM supplier, ( \
SELECT l_suppkey AS supplier_no, SUM(l_extendedprice * (1 - l_discount)) AS total_revenue \
FROM lineitem WHERE l_shipdate >= DATE '1996-01-01' AND l_shipdate < DATE '1996-04-01' \
GROUP BY l_suppkey) AS revenue0 \
WHERE s_suppkey = supplier_no AND total_revenue = ( \
SELECT MAX(total_revenue) FROM ( \
SELECT l_suppkey AS supplier_no, SUM(l_extendedprice * (1 - l_discount)) AS total_revenue \
FROM lineitem WHERE l_shipdate >= DATE '1996-01-01' AND l_shipdate < DATE '1996-04-01' \
GROUP BY l_suppkey) AS revenue1) \
ORDER BY s_suppkey";

const Q16: &str = "\
SELECT p_brand, p_type, p_size, COUNT(DISTINCT ps_suppkey) AS supplier_cnt \
FROM partsupp, part \
WHERE p_partkey = ps_partkey AND p_brand <> 'Brand#45' \
AND p_type NOT LIKE 'MEDIUM POLISHED%' AND p_size IN (49, 14, 23, 45, 19, 3, 36, 9) \
AND ps_suppkey NOT IN (SELECT s_suppkey FROM supplier WHERE s_comment LIKE '%Customer%Complaints%') \
GROUP BY p_brand, p_type, p_size ORDER BY supplier_cnt DESC, p_brand, p_type, p_size";

const Q17: &str = "\
SELECT SUM(l_extendedprice) / 7.0 AS avg_yearly FROM lineitem, part \
WHERE p_partkey = l_partkey AND p_brand = 'Brand#23' AND p_container = 'MED BOX' \
AND l_quantity < (SELECT 0.2 * AVG(l_quantity) FROM lineitem WHERE l_partkey = p_partkey)";

const Q18: &str = "\
SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, SUM(l_quantity) \
FROM customer, orders, lineitem \
WHERE o_orderkey IN (SELECT l_orderkey FROM lineitem GROUP BY l_orderkey HAVING SUM(l_quantity) > 300) \
AND c_custkey = o_custkey AND o_orderkey = l_orderkey \
GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice \
ORDER BY o_totalprice DESC, o_orderdate LIMIT 100";

const Q19: &str = "\
SELECT SUM(l_extendedprice * (1 - l_discount)) AS revenue FROM lineitem, part WHERE \
(p_partkey = l_partkey AND p_brand = 'Brand#12' \
AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG') \
AND l_quantity >= 1 AND l_quantity <= 11 AND p_size >= 1 AND p_size <= 5 \
AND l_shipmode IN ('AIR', 'REG AIR') AND l_shipinstruct = 'DELIVER IN PERSON') \
OR (p_partkey = l_partkey AND p_brand = 'Brand#23' \
AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK') \
AND l_quantity >= 10 AND l_quantity <= 20 AND p_size >= 1 AND p_size <= 10 \
AND l_shipmode IN ('AIR', 'REG AIR') AND l_shipinstruct = 'DELIVER IN PERSON') \
OR (p_partkey = l_partkey AND p_brand = 'Brand#34' \
AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG') \
AND l_quantity >= 20 AND l_quantity <= 30 AND p_size >= 1 AND p_size <= 15 \
AND l_shipmode IN ('AIR', 'REG AIR') AND l_shipinstruct = 'DELIVER IN PERSON')";

const Q20: &str = "\
SELECT s_name, s_address FROM supplier, nation \
WHERE s_suppkey IN (SELECT ps_suppkey FROM partsupp \
WHERE ps_partkey IN (SELECT p_partkey FROM part WHERE p_name LIKE 'forest%') \
AND ps_availqty > (SELECT 0.5 * SUM(l_quantity) FROM lineitem \
WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey \
AND l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01')) \
AND s_nationkey = n_nationkey AND n_name = 'CANADA' ORDER BY s_name";

const Q21: &str = "\
SELECT s_name, COUNT(*) AS numwait FROM supplier, lineitem l1, orders, nation \
WHERE s_suppkey = l1.l_suppkey AND o_orderkey = l1.l_orderkey AND o_orderstatus = 'F' \
AND l1.l_receiptdate > l1.l_commitdate \
AND EXISTS (SELECT * FROM lineitem l2 WHERE l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey) \
AND NOT EXISTS (SELECT * FROM lineitem l3 WHERE l3.l_orderkey = l1.l_orderkey \
AND l3.l_suppkey <> l1.l_suppkey AND l3.l_receiptdate > l3.l_commitdate) \
AND s_nationkey = n_nationkey AND n_name = 'SAUDI ARABIA' \
GROUP BY s_name ORDER BY numwait DESC, s_name LIMIT 100";

const Q22: &str = "\
SELECT cntrycode, COUNT(*) AS numcust, SUM(c_acctbal) AS totacctbal FROM ( \
SELECT SUBSTRING(c_phone FROM 1 FOR 2) AS cntrycode, c_acctbal FROM customer \
WHERE SUBSTRING(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17') \
AND c_acctbal > (SELECT AVG(c_acctbal) FROM customer \
WHERE c_acctbal > 0.00 AND SUBSTRING(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17')) \
AND NOT EXISTS (SELECT * FROM orders WHERE o_custkey = c_custkey)) AS custsale \
GROUP BY cntrycode ORDER BY cntrycode";

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

    /// Counts the rows in a statement by its top-level value groups.
    fn rows_in(statement: &str) -> usize {
        let Some(values) = statement.split_once(" VALUES ") else {
            return 0;
        };
        let mut depth = 0usize;
        let mut rows = 0usize;
        let mut in_string = false;
        let mut chars = values.1.chars().peekable();
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

    #[test]
    fn test_generation_is_reproducible_for_a_scale_factor() {
        let a = collect(0.001, 50);
        let b = collect(0.001, 50);
        assert_eq!(a, b, "the same scale factor produces the same database");
        assert!(!a.is_empty());
    }

    #[test]
    fn test_row_counts_match_what_the_scale_factor_promises() {
        let scale = 0.001;
        let statements = collect(scale, 100);
        let counts = counts_by_table(&statements);

        assert_eq!(counts["region"], 5, "region is fixed at five rows");
        assert_eq!(counts["nation"], 25, "nation is fixed at twenty five rows");
        assert_eq!(counts["part"], 200);
        assert_eq!(counts["supplier"], 10);
        assert_eq!(counts["customer"], 150);
        assert_eq!(
            counts["partsupp"],
            counts["part"] * 4,
            "four suppliers for every part"
        );
        assert_eq!(counts["orders"], 1500);
        // One to seven lines per order
        assert!(
            counts["lineitem"] >= counts["orders"] && counts["lineitem"] <= counts["orders"] * 7,
            "lineitem {} is outside one to seven per order ({})",
            counts["lineitem"],
            counts["orders"]
        );
    }

    #[test]
    fn test_the_promised_counts_match_the_generated_ones() {
        let scale = 0.001;
        let counts = counts_by_table(&collect(scale, 100));
        for (table, promised) in row_counts(scale) {
            // lineitem is an average, the rest are exact
            if table == "lineitem" {
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
    fn test_a_zero_or_negative_scale_is_refused() {
        let mut sink = |_: &str| Ok(());
        assert!(generate(0.0, 10, &mut sink).is_err());
        assert!(generate(-1.0, 10, &mut sink).is_err());
    }

    /// The query set selects on generated text. A generator whose vocabulary
    /// drifted from the queries would leave these predicates matching nothing
    /// and the run would report an empty workload as a fast one.
    #[test]
    fn test_the_generated_text_matches_what_the_queries_select_on() {
        let statements = collect(0.01, 500);
        let parts: String = statements
            .iter()
            .filter(|s| s.starts_with("INSERT INTO part "))
            .cloned()
            .collect();
        assert!(parts.contains("green"), "Q9 selects p_name like '%green%'");
        assert!(
            parts.contains("PROMO"),
            "Q14 splits on p_type like 'PROMO%'"
        );
        assert!(parts.contains("BRASS"), "Q2 selects p_type like '%BRASS'");
        assert!(
            parts.contains("MED BOX"),
            "Q17 selects p_container = 'MED BOX'"
        );

        let orders: String = statements
            .iter()
            .filter(|s| s.starts_with("INSERT INTO orders "))
            .cloned()
            .collect();
        assert!(
            orders.contains("special") && orders.contains("requests"),
            "Q13 counts orders whose comment does not match '%special%requests%'"
        );
    }

    /// Q22 reads the country code out of the first two characters of the
    /// phone number, so the encoding has to put the nation key there.
    #[test]
    fn test_a_phone_number_carries_its_nation_in_the_first_two_characters() {
        let mut rng = Rng::new(1);
        for nation in 0..25i64 {
            let p = phone(&mut rng, nation);
            let code: i64 = p[..2].parse().expect("two leading digits");
            assert_eq!(code, nation + 10);
            assert_eq!(p.len(), 15, "the column is VARCHAR(15)");
        }
    }

    #[test]
    fn test_generated_text_stays_inside_its_declared_column_width() {
        let mut rng = Rng::new(3);
        for width in [23usize, 44, 79, 101, 117, 152, 199] {
            for _ in 0..64 {
                let c = comment(&mut rng, width);
                assert!(
                    c.len() <= width,
                    "a {width} character column got {} characters",
                    c.len()
                );
            }
        }
    }

    #[test]
    fn test_every_query_is_present_and_named() {
        let qs = queries();
        assert_eq!(qs.len(), 22, "TPC-H is twenty two queries");
        for (i, (name, sql)) in qs.iter().enumerate() {
            assert_eq!(*name, format!("Q{}", i + 1));
            assert!(!sql.trim().is_empty(), "{name} has no text");
            assert!(
                sql.starts_with("SELECT"),
                "{name} does not start with SELECT"
            );
        }
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
}
