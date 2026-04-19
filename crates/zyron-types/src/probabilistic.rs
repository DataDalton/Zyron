//! Probabilistic data structures: HyperLogLog, Bloom Filter, T-Digest, Count-Min Sketch.
//!
//! All structures are serialized to bytes so they can be stored as column values
//! and merged across rows.

use zyron_common::hash64;
use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// HyperLogLog
// ---------------------------------------------------------------------------

// Tags to identify encoded sketch types
const HLL_TAG: u8 = 0x48;
const BLOOM_TAG: u8 = 0x42;
const TDIGEST_TAG: u8 = 0x54;
const CMS_TAG: u8 = 0x43;

/// Creates an empty HyperLogLog sketch with the given precision (4-16).
/// Precision 14 uses 16KB and gives ~0.8% standard error.
pub fn hll_create(precision: u8) -> Result<Vec<u8>> {
    if !(4..=16).contains(&precision) {
        return Err(ZyronError::ExecutionError(format!(
            "HLL precision must be 4..=16, got {}",
            precision
        )));
    }
    let m = 1usize << precision;
    let mut bytes = Vec::with_capacity(2 + m);
    bytes.push(HLL_TAG);
    bytes.push(precision);
    bytes.extend(std::iter::repeat(0u8).take(m));
    Ok(bytes)
}

fn hll_validate(sketch: &[u8]) -> Result<(u8, usize)> {
    if sketch.len() < 2 || sketch[0] != HLL_TAG {
        return Err(ZyronError::ExecutionError("Invalid HLL sketch".into()));
    }
    let precision = sketch[1];
    let m = 1usize << precision;
    if sketch.len() != 2 + m {
        return Err(ZyronError::ExecutionError(format!(
            "HLL size mismatch: expected {}, got {}",
            2 + m,
            sketch.len()
        )));
    }
    Ok((precision, m))
}

/// Adds an element to the HyperLogLog sketch.
pub fn hll_add(sketch: &mut [u8], value: &[u8]) -> Result<()> {
    let (precision, _m) = hll_validate(sketch)?;

    let hash = hash64(value);
    let p = precision as u32;
    let bucket = (hash >> (64 - p)) as usize;
    // Count leading zeros of the remaining bits (after the bucket bits)
    let rest = (hash << p) | (1u64 << (p.saturating_sub(1)));
    let leading_zeros = rest.leading_zeros() + 1;
    let rank = leading_zeros.min(64 - p + 1) as u8;

    if sketch[2 + bucket] < rank {
        sketch[2 + bucket] = rank;
    }
    Ok(())
}

/// Estimates the cardinality represented by the HyperLogLog sketch.
pub fn hll_count(sketch: &[u8]) -> Result<u64> {
    let (precision, m) = hll_validate(sketch)?;
    let m_f = m as f64;
    let alpha = match precision {
        4 => 0.673,
        5 => 0.697,
        6 => 0.709,
        _ => 0.7213 / (1.0 + 1.079 / m_f),
    };

    let mut sum = 0.0;
    let mut zeros = 0;
    for &r in &sketch[2..] {
        sum += 2f64.powi(-(r as i32));
        if r == 0 {
            zeros += 1;
        }
    }

    let raw_estimate = alpha * m_f * m_f / sum;

    // Small range correction (linear counting)
    if raw_estimate <= 2.5 * m_f && zeros > 0 {
        Ok((m_f * (m_f / zeros as f64).ln()).round() as u64)
    } else {
        Ok(raw_estimate.round() as u64)
    }
}

/// Merges two HLL sketches (union). Must have same precision.
pub fn hll_merge(a: &[u8], b: &[u8]) -> Result<Vec<u8>> {
    let (a_prec, a_m) = hll_validate(a)?;
    let (b_prec, _) = hll_validate(b)?;
    if a_prec != b_prec {
        return Err(ZyronError::ExecutionError(
            "Cannot merge HLLs with different precisions".into(),
        ));
    }
    let mut result = Vec::with_capacity(a.len());
    result.push(HLL_TAG);
    result.push(a_prec);
    for i in 0..a_m {
        result.push(a[2 + i].max(b[2 + i]));
    }
    Ok(result)
}

/// Returns the estimated relative error for the HLL sketch.
pub fn hll_error(sketch: &[u8]) -> Result<f64> {
    let (_prec, m) = hll_validate(sketch)?;
    Ok(1.04 / (m as f64).sqrt())
}

// ---------------------------------------------------------------------------
// Bloom Filter
// ---------------------------------------------------------------------------

/// Creates a Bloom filter sized for the expected item count and false positive rate.
/// Format: [BLOOM_TAG][u32 hash_count LE][u32 bit_count LE][bit_array]
pub fn bloom_create(expected_items: u64, false_positive_rate: f64) -> Result<Vec<u8>> {
    if expected_items == 0 {
        return Err(ZyronError::ExecutionError(
            "expected_items must be > 0".into(),
        ));
    }
    let fpr = false_positive_rate.clamp(1e-10, 0.5);

    // Optimal size: m = -n*ln(p) / (ln(2))^2
    let n = expected_items as f64;
    let m_bits = (-n * fpr.ln() / (2f64.ln()).powi(2)).ceil() as u64;
    let m_bits = m_bits.max(64);

    // Optimal number of hashes: k = (m/n) * ln(2)
    let k = ((m_bits as f64 / n) * 2f64.ln()).round() as u32;
    let k = k.clamp(1, 30);

    let byte_count = ((m_bits + 7) / 8) as usize;
    let mut bytes = Vec::with_capacity(9 + byte_count);
    bytes.push(BLOOM_TAG);
    bytes.extend_from_slice(&k.to_le_bytes());
    bytes.extend_from_slice(&(m_bits as u32).to_le_bytes());
    bytes.extend(std::iter::repeat(0u8).take(byte_count));
    Ok(bytes)
}

fn bloom_validate(filter: &[u8]) -> Result<(u32, u32)> {
    if filter.len() < 9 || filter[0] != BLOOM_TAG {
        return Err(ZyronError::ExecutionError("Invalid Bloom filter".into()));
    }
    let k = u32::from_le_bytes([filter[1], filter[2], filter[3], filter[4]]);
    let m = u32::from_le_bytes([filter[5], filter[6], filter[7], filter[8]]);
    let expected_bytes = 9 + ((m as usize + 7) / 8);
    if filter.len() != expected_bytes {
        return Err(ZyronError::ExecutionError(
            "Bloom filter size mismatch".into(),
        ));
    }
    Ok((k, m))
}

/// Adds an element to the Bloom filter.
pub fn bloom_add(filter: &mut [u8], value: &[u8]) -> Result<()> {
    let (k, m) = bloom_validate(filter)?;
    let hash = hash64(value);
    let hash2 = hash64(&hash.to_le_bytes());

    for i in 0..k {
        let combined = hash.wrapping_add((i as u64).wrapping_mul(hash2));
        let bit = (combined % m as u64) as usize;
        filter[9 + bit / 8] |= 1 << (bit % 8);
    }
    Ok(())
}

/// Returns true if the value may be in the set (false positives possible).
pub fn bloom_contains(filter: &[u8], value: &[u8]) -> Result<bool> {
    let (k, m) = bloom_validate(filter)?;
    let hash = hash64(value);
    let hash2 = hash64(&hash.to_le_bytes());

    for i in 0..k {
        let combined = hash.wrapping_add((i as u64).wrapping_mul(hash2));
        let bit = (combined % m as u64) as usize;
        if filter[9 + bit / 8] & (1 << (bit % 8)) == 0 {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Merges two Bloom filters (union). Must have same parameters.
pub fn bloom_merge(a: &[u8], b: &[u8]) -> Result<Vec<u8>> {
    let (a_k, a_m) = bloom_validate(a)?;
    let (b_k, b_m) = bloom_validate(b)?;
    if a_k != b_k || a_m != b_m {
        return Err(ZyronError::ExecutionError(
            "Cannot merge Bloom filters with different parameters".into(),
        ));
    }
    let mut result = a.to_vec();
    for i in 9..result.len() {
        result[i] |= b[i];
    }
    Ok(result)
}

/// Returns the current false positive rate given the set bit ratio.
pub fn bloom_false_positive_rate(filter: &[u8]) -> Result<f64> {
    let (k, m) = bloom_validate(filter)?;
    let mut set_bits: u64 = 0;
    for &byte in &filter[9..] {
        set_bits += byte.count_ones() as u64;
    }
    let ratio = set_bits as f64 / m as f64;
    Ok(ratio.powi(k as i32))
}

// ---------------------------------------------------------------------------
// T-Digest (simplified implementation)
// ---------------------------------------------------------------------------

/// Creates an empty T-Digest with the given compression parameter.
/// Format: [TDIGEST_TAG][u32 compression LE][u32 centroid_count LE][(f64 mean, u32 weight)*]
pub fn tdigest_create(compression: f64) -> Result<Vec<u8>> {
    if compression < 1.0 || compression > 10000.0 {
        return Err(ZyronError::ExecutionError(
            "compression must be in [1, 10000]".into(),
        ));
    }
    let compression = compression as u32;
    let mut bytes = Vec::with_capacity(9);
    bytes.push(TDIGEST_TAG);
    bytes.extend_from_slice(&compression.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes()); // centroid count
    Ok(bytes)
}

fn tdigest_parse(digest: &[u8]) -> Result<(u32, Vec<(f64, u32)>)> {
    if digest.len() < 9 || digest[0] != TDIGEST_TAG {
        return Err(ZyronError::ExecutionError("Invalid T-Digest".into()));
    }
    let compression = u32::from_le_bytes([digest[1], digest[2], digest[3], digest[4]]);
    let count = u32::from_le_bytes([digest[5], digest[6], digest[7], digest[8]]) as usize;
    let expected = 9 + count * 12;
    if digest.len() != expected {
        return Err(ZyronError::ExecutionError("T-Digest size mismatch".into()));
    }
    let mut centroids = Vec::with_capacity(count);
    for i in 0..count {
        let offset = 9 + i * 12;
        let mean = f64::from_le_bytes([
            digest[offset],
            digest[offset + 1],
            digest[offset + 2],
            digest[offset + 3],
            digest[offset + 4],
            digest[offset + 5],
            digest[offset + 6],
            digest[offset + 7],
        ]);
        let weight = u32::from_le_bytes([
            digest[offset + 8],
            digest[offset + 9],
            digest[offset + 10],
            digest[offset + 11],
        ]);
        centroids.push((mean, weight));
    }
    Ok((compression, centroids))
}

fn tdigest_serialize(compression: u32, centroids: &[(f64, u32)]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(9 + centroids.len() * 12);
    bytes.push(TDIGEST_TAG);
    bytes.extend_from_slice(&compression.to_le_bytes());
    bytes.extend_from_slice(&(centroids.len() as u32).to_le_bytes());
    for &(mean, weight) in centroids {
        bytes.extend_from_slice(&mean.to_le_bytes());
        bytes.extend_from_slice(&weight.to_le_bytes());
    }
    bytes
}

/// Adds a value to the T-Digest.
pub fn tdigest_add(digest: &mut Vec<u8>, value: f64) -> Result<()> {
    let (compression, mut centroids) = tdigest_parse(digest)?;

    // Insert into sorted order
    let insert_pos = centroids
        .iter()
        .position(|&(m, _)| m > value)
        .unwrap_or(centroids.len());
    centroids.insert(insert_pos, (value, 1));

    // Compress if too many centroids
    if centroids.len() > compression as usize * 2 {
        centroids = compress_centroids(&centroids, compression);
    }

    *digest = tdigest_serialize(compression, &centroids);
    Ok(())
}

fn compress_centroids(centroids: &[(f64, u32)], compression: u32) -> Vec<(f64, u32)> {
    let total_weight: u64 = centroids.iter().map(|&(_, w)| w as u64).sum();
    let max_weight = (total_weight / compression as u64).max(1) as u32;

    let mut result: Vec<(f64, u32)> = Vec::new();
    for &(mean, weight) in centroids {
        if let Some(last) = result.last_mut() {
            if last.1 + weight <= max_weight {
                let combined_weight = last.1 + weight;
                let new_mean =
                    (last.0 * last.1 as f64 + mean * weight as f64) / combined_weight as f64;
                *last = (new_mean, combined_weight);
                continue;
            }
        }
        result.push((mean, weight));
    }
    result
}

/// Returns the estimated value at the given quantile (0.0 to 1.0).
pub fn tdigest_quantile(digest: &[u8], q: f64) -> Result<f64> {
    let (_compression, centroids) = tdigest_parse(digest)?;
    if centroids.is_empty() {
        return Err(ZyronError::ExecutionError("Empty T-Digest".into()));
    }
    let q = q.clamp(0.0, 1.0);
    let total_weight: u64 = centroids.iter().map(|&(_, w)| w as u64).sum();
    let target = q * total_weight as f64;

    let mut cumulative = 0.0;
    for &(mean, weight) in &centroids {
        let center = cumulative + weight as f64 / 2.0;
        if target <= center {
            return Ok(mean);
        }
        cumulative += weight as f64;
    }

    Ok(centroids[centroids.len() - 1].0)
}

/// Merges two T-Digests.
pub fn tdigest_merge(a: &[u8], b: &[u8]) -> Result<Vec<u8>> {
    let (a_comp, a_centroids) = tdigest_parse(a)?;
    let (b_comp, b_centroids) = tdigest_parse(b)?;
    let compression = a_comp.max(b_comp);

    let mut merged: Vec<(f64, u32)> = Vec::with_capacity(a_centroids.len() + b_centroids.len());
    merged.extend_from_slice(&a_centroids);
    merged.extend_from_slice(&b_centroids);
    merged.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    let compressed = compress_centroids(&merged, compression);

    Ok(tdigest_serialize(compression, &compressed))
}

/// Returns the fraction of values <= the given value (cumulative distribution).
pub fn tdigest_cdf(digest: &[u8], value: f64) -> Result<f64> {
    let (_compression, centroids) = tdigest_parse(digest)?;
    if centroids.is_empty() {
        return Ok(0.0);
    }
    let total_weight: u64 = centroids.iter().map(|&(_, w)| w as u64).sum();
    let mut cumulative = 0u64;
    for &(mean, weight) in &centroids {
        if value < mean {
            return Ok(cumulative as f64 / total_weight as f64);
        }
        cumulative += weight as u64;
    }
    Ok(1.0)
}

// ---------------------------------------------------------------------------
// Count-Min Sketch
// ---------------------------------------------------------------------------

/// Creates an empty Count-Min Sketch.
/// Format: [CMS_TAG][u32 width LE][u32 depth LE][u64 counters...]
pub fn cms_create(width: u32, depth: u32) -> Result<Vec<u8>> {
    if width == 0 || depth == 0 {
        return Err(ZyronError::ExecutionError(
            "CMS width and depth must be > 0".into(),
        ));
    }
    if depth > 16 {
        return Err(ZyronError::ExecutionError(
            "CMS depth > 16 not supported".into(),
        ));
    }
    let counter_count = (width as usize) * (depth as usize);
    let mut bytes = Vec::with_capacity(9 + counter_count * 8);
    bytes.push(CMS_TAG);
    bytes.extend_from_slice(&width.to_le_bytes());
    bytes.extend_from_slice(&depth.to_le_bytes());
    bytes.extend(std::iter::repeat(0u8).take(counter_count * 8));
    Ok(bytes)
}

fn cms_validate(sketch: &[u8]) -> Result<(u32, u32)> {
    if sketch.len() < 9 || sketch[0] != CMS_TAG {
        return Err(ZyronError::ExecutionError("Invalid CMS".into()));
    }
    let width = u32::from_le_bytes([sketch[1], sketch[2], sketch[3], sketch[4]]);
    let depth = u32::from_le_bytes([sketch[5], sketch[6], sketch[7], sketch[8]]);
    let expected = 9 + (width as usize) * (depth as usize) * 8;
    if sketch.len() != expected {
        return Err(ZyronError::ExecutionError("CMS size mismatch".into()));
    }
    Ok((width, depth))
}

fn cms_index(value: &[u8], row: u32, width: u32) -> usize {
    let seed = row as u64;
    let h = hash64(value).wrapping_add(seed.wrapping_mul(0x9E3779B97F4A7C15));
    (h % width as u64) as usize
}

/// Adds a value to the CMS with the given count.
pub fn cms_add(sketch: &mut [u8], value: &[u8], count: u64) -> Result<()> {
    let (width, depth) = cms_validate(sketch)?;
    for row in 0..depth {
        let col = cms_index(value, row, width);
        let offset = 9 + (row as usize * width as usize + col) * 8;
        let current = u64::from_le_bytes([
            sketch[offset],
            sketch[offset + 1],
            sketch[offset + 2],
            sketch[offset + 3],
            sketch[offset + 4],
            sketch[offset + 5],
            sketch[offset + 6],
            sketch[offset + 7],
        ]);
        let new_val = current.saturating_add(count);
        sketch[offset..offset + 8].copy_from_slice(&new_val.to_le_bytes());
    }
    Ok(())
}

/// Returns the estimated count for the value.
pub fn cms_estimate(sketch: &[u8], value: &[u8]) -> Result<u64> {
    let (width, depth) = cms_validate(sketch)?;
    let mut min_count = u64::MAX;
    for row in 0..depth {
        let col = cms_index(value, row, width);
        let offset = 9 + (row as usize * width as usize + col) * 8;
        let current = u64::from_le_bytes([
            sketch[offset],
            sketch[offset + 1],
            sketch[offset + 2],
            sketch[offset + 3],
            sketch[offset + 4],
            sketch[offset + 5],
            sketch[offset + 6],
            sketch[offset + 7],
        ]);
        if current < min_count {
            min_count = current;
        }
    }
    Ok(min_count)
}

/// Merges two CMS sketches. Must have same dimensions.
pub fn cms_merge(a: &[u8], b: &[u8]) -> Result<Vec<u8>> {
    let (a_w, a_d) = cms_validate(a)?;
    let (b_w, b_d) = cms_validate(b)?;
    if a_w != b_w || a_d != b_d {
        return Err(ZyronError::ExecutionError(
            "Cannot merge CMS with different dimensions".into(),
        ));
    }
    let mut result = a.to_vec();
    let counter_count = (a_w as usize) * (a_d as usize);
    for i in 0..counter_count {
        let offset = 9 + i * 8;
        let av = u64::from_le_bytes([
            a[offset],
            a[offset + 1],
            a[offset + 2],
            a[offset + 3],
            a[offset + 4],
            a[offset + 5],
            a[offset + 6],
            a[offset + 7],
        ]);
        let bv = u64::from_le_bytes([
            b[offset],
            b[offset + 1],
            b[offset + 2],
            b[offset + 3],
            b[offset + 4],
            b[offset + 5],
            b[offset + 6],
            b[offset + 7],
        ]);
        let sum = av.saturating_add(bv);
        result[offset..offset + 8].copy_from_slice(&sum.to_le_bytes());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    // HLL tests
    #[test]
    fn test_hll_create() {
        let sketch = hll_create(14).unwrap();
        assert_eq!(sketch[0], HLL_TAG);
        assert_eq!(sketch[1], 14);
        assert_eq!(sketch.len(), 2 + (1 << 14));
    }

    #[test]
    fn test_hll_invalid_precision() {
        assert!(hll_create(3).is_err());
        assert!(hll_create(17).is_err());
    }

    #[test]
    fn test_hll_count_empty() {
        let sketch = hll_create(14).unwrap();
        assert_eq!(hll_count(&sketch).unwrap(), 0);
    }

    #[test]
    fn test_hll_count_single() {
        let mut sketch = hll_create(14).unwrap();
        hll_add(&mut sketch, b"test").unwrap();
        let count = hll_count(&sketch).unwrap();
        assert!(count >= 1 && count <= 2);
    }

    #[test]
    fn test_hll_count_duplicates() {
        let mut sketch = hll_create(14).unwrap();
        for _ in 0..1000 {
            hll_add(&mut sketch, b"same_value").unwrap();
        }
        let count = hll_count(&sketch).unwrap();
        assert!(count <= 3); // should estimate ~1
    }

    #[test]
    fn test_hll_count_many_unique() {
        let mut sketch = hll_create(14).unwrap();
        for i in 0..10000u32 {
            hll_add(&mut sketch, &i.to_le_bytes()).unwrap();
        }
        let count = hll_count(&sketch).unwrap();
        // Should be within ~2% error
        let diff = (count as i64 - 10000).abs();
        assert!(diff < 300, "Expected ~10000, got {}", count);
    }

    #[test]
    fn test_hll_merge() {
        let mut a = hll_create(10).unwrap();
        let mut b = hll_create(10).unwrap();
        for i in 0..100u32 {
            hll_add(&mut a, &i.to_le_bytes()).unwrap();
            hll_add(&mut b, &(i + 50).to_le_bytes()).unwrap();
        }
        let merged = hll_merge(&a, &b).unwrap();
        let count = hll_count(&merged).unwrap();
        // Union has 150 unique values
        assert!(count > 100 && count < 200);
    }

    #[test]
    fn test_hll_error() {
        let sketch = hll_create(14).unwrap();
        let err = hll_error(&sketch).unwrap();
        // Error for p=14: 1.04/sqrt(16384) ~ 0.008 = 0.8%
        assert!(err > 0.005 && err < 0.02);
    }

    // Bloom filter tests
    #[test]
    fn test_bloom_create() {
        let filter = bloom_create(1000, 0.01).unwrap();
        assert_eq!(filter[0], BLOOM_TAG);
    }

    #[test]
    fn test_bloom_add_contains() {
        let mut filter = bloom_create(1000, 0.01).unwrap();
        bloom_add(&mut filter, b"hello").unwrap();
        assert!(bloom_contains(&filter, b"hello").unwrap());
    }

    #[test]
    fn test_bloom_no_false_negatives() {
        let mut filter = bloom_create(100, 0.01).unwrap();
        let items = ["apple", "banana", "cherry", "date"];
        for item in items {
            bloom_add(&mut filter, item.as_bytes()).unwrap();
        }
        for item in items {
            assert!(bloom_contains(&filter, item.as_bytes()).unwrap());
        }
    }

    #[test]
    fn test_bloom_absent_item() {
        let mut filter = bloom_create(1000, 0.001).unwrap();
        bloom_add(&mut filter, b"hello").unwrap();
        // "xyz" was not added - should return false with high probability
        // (no guarantee due to false positives)
        let _ = bloom_contains(&filter, b"xyz").unwrap();
    }

    #[test]
    fn test_bloom_merge() {
        let mut a = bloom_create(100, 0.01).unwrap();
        let mut b = bloom_create(100, 0.01).unwrap();
        bloom_add(&mut a, b"foo").unwrap();
        bloom_add(&mut b, b"bar").unwrap();
        let merged = bloom_merge(&a, &b).unwrap();
        assert!(bloom_contains(&merged, b"foo").unwrap());
        assert!(bloom_contains(&merged, b"bar").unwrap());
    }

    #[test]
    fn test_bloom_fpr() {
        let filter = bloom_create(100, 0.01).unwrap();
        // Empty filter should have fpr = 0
        assert_eq!(bloom_false_positive_rate(&filter).unwrap(), 0.0);
    }

    // T-Digest tests
    #[test]
    fn test_tdigest_create() {
        let digest = tdigest_create(100.0).unwrap();
        assert_eq!(digest[0], TDIGEST_TAG);
    }

    #[test]
    fn test_tdigest_add_quantile() {
        let mut digest = tdigest_create(100.0).unwrap();
        for i in 1..=100i32 {
            tdigest_add(&mut digest, i as f64).unwrap();
        }
        let median = tdigest_quantile(&digest, 0.5).unwrap();
        assert!((median - 50.5).abs() < 5.0);
        let p99 = tdigest_quantile(&digest, 0.99).unwrap();
        assert!(p99 > 90.0);
    }

    #[test]
    fn test_tdigest_merge() {
        let mut a = tdigest_create(100.0).unwrap();
        let mut b = tdigest_create(100.0).unwrap();
        for i in 0..50i32 {
            tdigest_add(&mut a, i as f64).unwrap();
        }
        for i in 50..100i32 {
            tdigest_add(&mut b, i as f64).unwrap();
        }
        let merged = tdigest_merge(&a, &b).unwrap();
        let median = tdigest_quantile(&merged, 0.5).unwrap();
        assert!((median - 50.0).abs() < 10.0);
    }

    #[test]
    fn test_tdigest_cdf() {
        let mut digest = tdigest_create(100.0).unwrap();
        for i in 1..=100i32 {
            tdigest_add(&mut digest, i as f64).unwrap();
        }
        let cdf_50 = tdigest_cdf(&digest, 50.0).unwrap();
        assert!((cdf_50 - 0.5).abs() < 0.1);
    }

    // CMS tests
    #[test]
    fn test_cms_create() {
        let sketch = cms_create(100, 5).unwrap();
        assert_eq!(sketch[0], CMS_TAG);
    }

    #[test]
    fn test_cms_add_estimate() {
        let mut sketch = cms_create(1000, 5).unwrap();
        cms_add(&mut sketch, b"apple", 10).unwrap();
        cms_add(&mut sketch, b"banana", 20).unwrap();
        let apple_count = cms_estimate(&sketch, b"apple").unwrap();
        let banana_count = cms_estimate(&sketch, b"banana").unwrap();
        assert!(apple_count >= 10);
        assert!(banana_count >= 20);
    }

    #[test]
    fn test_cms_never_underestimates() {
        let mut sketch = cms_create(100, 3).unwrap();
        for _ in 0..500 {
            cms_add(&mut sketch, b"item", 1).unwrap();
        }
        let estimate = cms_estimate(&sketch, b"item").unwrap();
        assert!(estimate >= 500);
    }

    #[test]
    fn test_cms_merge() {
        let mut a = cms_create(100, 3).unwrap();
        let mut b = cms_create(100, 3).unwrap();
        cms_add(&mut a, b"x", 5).unwrap();
        cms_add(&mut b, b"x", 10).unwrap();
        let merged = cms_merge(&a, &b).unwrap();
        let count = cms_estimate(&merged, b"x").unwrap();
        assert!(count >= 15);
    }

    #[test]
    fn test_cms_invalid_dimensions() {
        assert!(cms_create(0, 5).is_err());
        assert!(cms_create(100, 0).is_err());
    }
}
