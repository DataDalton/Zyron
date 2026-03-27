#![allow(non_snake_case)]
//! TPC-H and TPC-C benchmark scaffolds.
//!
//! Creates the standard schema via the wire protocol and prints a message
//! indicating that data generation is not yet implemented.

use super::remote::RemoteClient;

/// Creates the TPC-H schema and prints status. Data generation and query
/// execution are not yet implemented.
pub fn runTpch(client: &mut RemoteClient, scale: f64) -> Result<(), String> {
    println!("TPC-H benchmark at scale factor {}", scale);
    println!("Creating schema...");

    let tables = [
        "CREATE TABLE IF NOT EXISTS nation (n_nationkey INT, n_name VARCHAR(25), n_regionkey INT, n_comment VARCHAR(152))",
        "CREATE TABLE IF NOT EXISTS region (r_regionkey INT, r_name VARCHAR(25), r_comment VARCHAR(152))",
        "CREATE TABLE IF NOT EXISTS part (p_partkey INT, p_name VARCHAR(55), p_mfgr VARCHAR(25), p_brand VARCHAR(10), p_type VARCHAR(25), p_size INT, p_container VARCHAR(10), p_retailprice DECIMAL(15,2), p_comment VARCHAR(23))",
        "CREATE TABLE IF NOT EXISTS supplier (s_suppkey INT, s_name VARCHAR(25), s_address VARCHAR(40), s_nationkey INT, s_phone VARCHAR(15), s_acctbal DECIMAL(15,2), s_comment VARCHAR(101))",
        "CREATE TABLE IF NOT EXISTS partsupp (ps_partkey INT, ps_suppkey INT, ps_availqty INT, ps_supplycost DECIMAL(15,2), ps_comment VARCHAR(199))",
        "CREATE TABLE IF NOT EXISTS customer (c_custkey INT, c_name VARCHAR(25), c_address VARCHAR(40), c_nationkey INT, c_phone VARCHAR(15), c_acctbal DECIMAL(15,2), c_mktsegment VARCHAR(10), c_comment VARCHAR(117))",
        "CREATE TABLE IF NOT EXISTS orders (o_orderkey INT, o_custkey INT, o_orderstatus VARCHAR(1), o_totalprice DECIMAL(15,2), o_orderdate DATE, o_orderpriority VARCHAR(15), o_clerk VARCHAR(15), o_shippriority INT, o_comment VARCHAR(79))",
        "CREATE TABLE IF NOT EXISTS lineitem (l_orderkey INT, l_partkey INT, l_suppkey INT, l_linenumber INT, l_quantity DECIMAL(15,2), l_extendedprice DECIMAL(15,2), l_discount DECIMAL(15,2), l_tax DECIMAL(15,2), l_returnflag VARCHAR(1), l_linestatus VARCHAR(1), l_shipdate DATE, l_commitdate DATE, l_receiptdate DATE, l_shipinstruct VARCHAR(25), l_shipmode VARCHAR(10), l_comment VARCHAR(44))",
    ];

    for sql in &tables {
        if let Err(e) = client.execute(sql) {
            eprintln!("Schema creation failed: {}", e);
            return Err(e);
        }
    }

    println!("TPC-H schema created ({} tables).", tables.len());
    println!("TPC-H data generation and query execution not yet implemented.");
    println!("Use a TPC-H data generator to populate tables.");
    Ok(())
}

/// TPC-C benchmark scaffold. Not yet implemented.
pub fn runTpcc(_client: &mut RemoteClient, scale: f64) -> Result<(), String> {
    println!("TPC-C benchmark at scale factor {}", scale);
    println!("TPC-C benchmark not yet implemented.");
    Ok(())
}
