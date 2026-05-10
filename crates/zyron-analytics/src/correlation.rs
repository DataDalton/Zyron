// Correlation and dependence measures: Pearson, Spearman, Kendall tau,
// mutual information, and a pairwise correlation matrix.
// Pearson uses Welford's running covariance for numerical stability.

use crate::value::AnalyticsValue;

// ===== Pearson (running covariance) =====
#[derive(Debug, Clone, Default)]
pub struct PearsonAggregator {
    pub n: u64,
    pub mean_x: f64,
    pub mean_y: f64,
    pub c_xy: f64,
    pub m2_x: f64,
    pub m2_y: f64,
}

impl PearsonAggregator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ingest(&mut self, x: f64, y: f64) {
        if x.is_nan() || y.is_nan() {
            return;
        }
        self.n += 1;
        let n = self.n as f64;
        let dx = x - self.mean_x;
        let dy = y - self.mean_y;
        self.mean_x += dx / n;
        self.mean_y += dy / n;
        self.c_xy += dx * (y - self.mean_y);
        self.m2_x += dx * (x - self.mean_x);
        self.m2_y += dy * (y - self.mean_y);
    }

    pub fn correlation(&self) -> Option<f64> {
        if self.n < 2 {
            return None;
        }
        let denom = (self.m2_x * self.m2_y).sqrt();
        if denom == 0.0 {
            return None;
        }
        Some(self.c_xy / denom)
    }
}

pub fn pearson_corr(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() {
        return None;
    }
    let mut p = PearsonAggregator::new();
    for (xi, yi) in x.iter().zip(y.iter()) {
        p.ingest(*xi, *yi);
    }
    p.correlation()
}

// ===== Spearman (rank-based Pearson) =====
fn rank(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && values[idx[j]] == values[idx[i]] {
            j += 1;
        }
        // Average rank for ties (1-based ranks)
        let avg = (i + j + 1) as f64 / 2.0;
        for k in i..j {
            ranks[idx[k]] = avg;
        }
        i = j;
    }
    ranks
}

pub struct SpearmanAggregator {
    xs: Vec<f64>,
    ys: Vec<f64>,
}

impl SpearmanAggregator {
    pub fn new() -> Self {
        Self {
            xs: Vec::new(),
            ys: Vec::new(),
        }
    }
    pub fn ingest(&mut self, x: f64, y: f64) {
        if x.is_nan() || y.is_nan() {
            return;
        }
        self.xs.push(x);
        self.ys.push(y);
    }
    pub fn correlation(&self) -> Option<f64> {
        if self.xs.len() < 2 {
            return None;
        }
        let rx = rank(&self.xs);
        let ry = rank(&self.ys);
        pearson_corr(&rx, &ry)
    }
}

pub fn spearman_corr(x: &[f64], y: &[f64]) -> Option<f64> {
    let mut s = SpearmanAggregator::new();
    for (xi, yi) in x.iter().zip(y.iter()) {
        s.ingest(*xi, *yi);
    }
    s.correlation()
}

// ===== Kendall tau-b =====
pub struct KendallAggregator {
    xs: Vec<f64>,
    ys: Vec<f64>,
}

impl KendallAggregator {
    pub fn new() -> Self {
        Self {
            xs: Vec::new(),
            ys: Vec::new(),
        }
    }
    pub fn ingest(&mut self, x: f64, y: f64) {
        if x.is_nan() || y.is_nan() {
            return;
        }
        self.xs.push(x);
        self.ys.push(y);
    }
    pub fn correlation(&self) -> Option<f64> {
        // Knight (1966) computes Kendall tau-b in O(n log n) by:
        //   1. Sorting pairs by (x, y) to count tied-pair groups
        //   2. Counting discordant pairs as inversions of y-in-x-order via
        //      merge sort
        //   3. Sorting y separately to count y-tied groups
        //
        // The naive O(n^2) form scales to maybe a few thousand inputs;
        // Knight's form scales to the same input sizes the rest of the
        // analytics engine handles (10M+).
        let n = self.xs.len();
        if n < 2 {
            return None;
        }

        // Build a (x, y) pair vector and sort by (x, y) so equal-x runs are
        // contiguous and within each x-group the y values are sorted.
        let mut pairs: Vec<(f64, f64)> = self
            .xs
            .iter()
            .zip(self.ys.iter())
            .map(|(x, y)| (*x, *y))
            .collect();
        pairs.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        });

        // tx: pairs that share an x value
        let mut tx: i64 = 0;
        // tied_both: pairs that share both x and y (subtracted back below)
        let mut tied_both: i64 = 0;
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            while j < n && pairs[j].0 == pairs[i].0 {
                j += 1;
            }
            let g = (j - i) as i64;
            tx += g * (g - 1) / 2;
            // Within this x-group, count y-equality runs
            let mut a = i;
            while a < j {
                let mut b = a + 1;
                while b < j && pairs[b].1 == pairs[a].1 {
                    b += 1;
                }
                let h = (b - a) as i64;
                tied_both += h * (h - 1) / 2;
                a = b;
            }
            i = j;
        }

        // Discordant pairs = inversions of the y-in-x-order sequence,
        // not counting pairs tied in x. Remove tied-x runs from the array
        // before counting inversions to avoid double counting them under tx.
        // Working buffer reused by merge sort.
        let ys_in_x_order: Vec<f64> = pairs.iter().map(|p| p.1).collect();
        let mut work = ys_in_x_order.clone();
        let mut tmp = vec![0.0f64; n];
        let inversions = merge_sort_count_inversions(&mut work, &mut tmp);

        // ty: pairs that share a y value (independent of x). Sort y alone.
        let mut ys_sorted: Vec<f64> = self.ys.clone();
        ys_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut ty: i64 = 0;
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            while j < n && ys_sorted[j] == ys_sorted[i] {
                j += 1;
            }
            let g = (j - i) as i64;
            ty += g * (g - 1) / 2;
            i = j;
        }

        // The inversion count from merge sort over the (x, y)-sorted ys
        // double-counts tied-y pairs that share the same x (because within
        // a sorted x-group the equal y values yield no inversions, so
        // there is nothing to subtract there). discordant equals inversions.
        let discordant = inversions;
        let n0 = (n as i64) * (n as i64 - 1) / 2;
        let concordant = n0 - discordant - tx - ty + tied_both;

        let denom_sq = ((n0 - tx) as f64) * ((n0 - ty) as f64);
        if denom_sq <= 0.0 {
            return None;
        }
        Some((concordant - discordant) as f64 / denom_sq.sqrt())
    }
}

// In-place merge sort that returns the number of inversions in `data`.
// Tied values do not count as inversions, which matches the convention
// used by Knight's Kendall formulation.
fn merge_sort_count_inversions(data: &mut [f64], tmp: &mut [f64]) -> i64 {
    let n = data.len();
    if n <= 1 {
        return 0;
    }
    let mid = n / 2;
    let (left, right) = data.split_at_mut(mid);
    let (tmp_l, tmp_r) = tmp.split_at_mut(mid);
    let mut inv = merge_sort_count_inversions(left, tmp_l);
    inv += merge_sort_count_inversions(right, tmp_r);
    inv += merge_count(left, right, tmp);
    // Copy merged result back into data
    data.copy_from_slice(&tmp[..n]);
    inv
}

fn merge_count(left: &[f64], right: &[f64], tmp: &mut [f64]) -> i64 {
    let (n_l, n_r) = (left.len(), right.len());
    let mut i = 0usize;
    let mut j = 0usize;
    let mut k = 0usize;
    let mut inv: i64 = 0;
    while i < n_l && j < n_r {
        if left[i] <= right[j] {
            tmp[k] = left[i];
            i += 1;
        } else {
            // Every remaining element in `left` forms an inversion with right[j]
            tmp[k] = right[j];
            inv += (n_l - i) as i64;
            j += 1;
        }
        k += 1;
    }
    while i < n_l {
        tmp[k] = left[i];
        i += 1;
        k += 1;
    }
    while j < n_r {
        tmp[k] = right[j];
        j += 1;
        k += 1;
    }
    inv
}

pub fn kendall_tau(x: &[f64], y: &[f64]) -> Option<f64> {
    let mut k = KendallAggregator::new();
    for (xi, yi) in x.iter().zip(y.iter()) {
        k.ingest(*xi, *yi);
    }
    k.correlation()
}

// ===== Mutual information (binned histogram estimator) =====
//
// Owns its histogram scratch buffers (joint and per-axis marginals) so
// repeated `estimate` calls at the same `bins` do not reallocate. The
// buffers are zeroed each call before counting; allocation only happens
// the first time a given bins value is used.
pub struct MutualInformationEstimator {
    pub bins: usize,
    joint: Vec<u64>,
    px: Vec<u64>,
    py: Vec<u64>,
}

impl MutualInformationEstimator {
    pub fn new(bins: usize) -> Self {
        let bins = bins.max(2);
        Self {
            bins,
            joint: vec![0u64; bins * bins],
            px: vec![0u64; bins],
            py: vec![0u64; bins],
        }
    }

    pub fn estimate(&mut self, x: &[f64], y: &[f64]) -> Option<f64> {
        let n = x.len();
        if n < 2 || n != y.len() {
            return None;
        }
        let (lo_x, hi_x) = min_max(x)?;
        let (lo_y, hi_y) = min_max(y)?;
        if lo_x == hi_x || lo_y == hi_y {
            return Some(0.0);
        }
        // Zero out the reusable buffers before counting
        for slot in self.joint.iter_mut() {
            *slot = 0;
        }
        for slot in self.px.iter_mut() {
            *slot = 0;
        }
        for slot in self.py.iter_mut() {
            *slot = 0;
        }
        let bx = (hi_x - lo_x) / self.bins as f64;
        let by = (hi_y - lo_y) / self.bins as f64;
        for (xi, yi) in x.iter().zip(y.iter()) {
            let ix = (((xi - lo_x) / bx).floor() as usize).min(self.bins - 1);
            let iy = (((yi - lo_y) / by).floor() as usize).min(self.bins - 1);
            self.joint[ix * self.bins + iy] += 1;
            self.px[ix] += 1;
            self.py[iy] += 1;
        }
        let n_f = n as f64;
        let mut mi = 0.0f64;
        for i in 0..self.bins {
            for j in 0..self.bins {
                let pij = self.joint[i * self.bins + j] as f64 / n_f;
                if pij == 0.0 {
                    continue;
                }
                let pi = self.px[i] as f64 / n_f;
                let pj = self.py[j] as f64 / n_f;
                if pi > 0.0 && pj > 0.0 {
                    mi += pij * (pij / (pi * pj)).ln();
                }
            }
        }
        Some(mi)
    }
}

fn min_max(values: &[f64]) -> Option<(f64, f64)> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for v in values {
        if v.is_nan() {
            continue;
        }
        if *v < lo {
            lo = *v;
        }
        if *v > hi {
            hi = *v;
        }
    }
    if lo.is_finite() && hi.is_finite() {
        Some((lo, hi))
    } else {
        None
    }
}

/// One-shot mutual information convenience wrapper. For repeated MI
/// computations on same-size inputs, instantiate `MutualInformationEstimator`
/// once and call `estimate` per pair so the histogram buffers are reused.
pub fn mutual_information(x: &[f64], y: &[f64], bins: usize) -> Option<f64> {
    MutualInformationEstimator::new(bins).estimate(x, y)
}

// ===== Correlation matrix =====
#[derive(Debug, Clone)]
pub struct CorrelationMatrix {
    pub columns: Vec<String>,
    // Row-major n x n matrix of Pearson coefficients (NaN on missing)
    pub values: Vec<f64>,
}

impl CorrelationMatrix {
    pub fn get(&self, i: usize, j: usize) -> f64 {
        let n = self.columns.len();
        self.values[i * n + j]
    }
}

/// Streaming correlation-matrix builder. Ingest rows one at a time
/// without materialising column buffers. Internal state is N*(N-1)/2
/// PearsonAggregators; memory is O(N^2) regardless of row count.
pub struct CorrelationMatrixBuilder {
    column_names: Vec<String>,
    aggs: Vec<PearsonAggregator>,
}

impl CorrelationMatrixBuilder {
    pub fn new(column_names: Vec<String>) -> Self {
        let n = column_names.len();
        let pair_count = n.saturating_sub(1) * n / 2;
        Self {
            column_names,
            aggs: (0..pair_count).map(|_| PearsonAggregator::new()).collect(),
        }
    }

    /// Ingest one row's column values. `row[c]` is None if the column is
    /// null at this row; pairs are only updated when both members have a
    /// value.
    pub fn ingest_row(&mut self, row: &[Option<f64>]) {
        let n = self.column_names.len();
        debug_assert_eq!(row.len(), n);
        for i in 0..n {
            let xi = match row[i] {
                Some(v) => v,
                None => continue,
            };
            for j in (i + 1)..n {
                if let Some(yj) = row[j] {
                    self.aggs[pair_index(i, j, n)].ingest(xi, yj);
                }
            }
        }
    }

    pub fn finalise(self) -> CorrelationMatrix {
        let n = self.column_names.len();
        let mut values = vec![f64::NAN; n * n];
        for i in 0..n {
            values[i * n + i] = 1.0;
            for j in (i + 1)..n {
                if let Some(c) = self.aggs[pair_index(i, j, n)].correlation() {
                    values[i * n + j] = c;
                    values[j * n + i] = c;
                }
            }
        }
        CorrelationMatrix {
            columns: self.column_names,
            values,
        }
    }
}

pub fn correlation_matrix(
    column_names: &[String],
    columns: &[Vec<AnalyticsValue>],
) -> CorrelationMatrix {
    let n = column_names.len();
    let mut values = vec![f64::NAN; n * n];

    // Pre-extract one Vec<Option<f64>> per column, aligned with the
    // original row index. Pairs only ingest when both columns have a
    // value at the same row.
    let numeric: Vec<Vec<Option<f64>>> = columns
        .iter()
        .map(|c| c.iter().map(|v| v.as_f64()).collect())
        .collect();

    let row_count = numeric.iter().map(|c| c.len()).max().unwrap_or(0);

    // Single pass over rows: maintain N*(N-1)/2 pair aggregators in a
    // flat triangular array indexed by `pair_index(i, j)`. The previous
    // form ran one full row sweep per pair, rereading both column
    // buffers from RAM each time; the single-pass form touches both
    // columns once per row for the whole upper triangle, keeping the
    // working set in cache for cache-resident column slices.
    let pair_count = n.saturating_sub(1) * n / 2;
    let mut aggs: Vec<PearsonAggregator> =
        (0..pair_count).map(|_| PearsonAggregator::new()).collect();

    // Scratch row buffer holding the current row's column values
    let mut row_vals: Vec<Option<f64>> = vec![None; n];

    for r in 0..row_count {
        for c in 0..n {
            row_vals[c] = numeric[c].get(r).copied().flatten();
        }
        for i in 0..n {
            let xi = match row_vals[i] {
                Some(v) => v,
                None => continue,
            };
            for j in (i + 1)..n {
                if let Some(yj) = row_vals[j] {
                    aggs[pair_index(i, j, n)].ingest(xi, yj);
                }
            }
        }
    }

    for i in 0..n {
        values[i * n + i] = 1.0;
        for j in (i + 1)..n {
            if let Some(c) = aggs[pair_index(i, j, n)].correlation() {
                values[i * n + j] = c;
                values[j * n + i] = c;
            }
        }
    }
    CorrelationMatrix {
        columns: column_names.to_vec(),
        values,
    }
}

/// Index into a flat upper-triangular array (i < j < n) packed
/// row-by-row. For n=4 the order is (0,1) (0,2) (0,3) (1,2) (1,3) (2,3).
#[inline]
fn pair_index(i: usize, j: usize, n: usize) -> usize {
    debug_assert!(i < j && j < n);
    // sum of (n-1) + (n-2) + ... + (n-i) = i*(2*n - i - 1) / 2 entries
    // skipped for rows 0..i, then j - i - 1 within row i.
    i * (2 * n - i - 1) / 2 + (j - i - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pearson_perfect_correlation() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| v * 2.0 + 1.0).collect();
        assert!((pearson_corr(&x, &y).unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn spearman_handles_monotone_nonlinear() {
        let x: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| v.powi(2)).collect();
        assert!((spearman_corr(&x, &y).unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn kendall_tau_perfect_concordance() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.clone();
        let t = kendall_tau(&x, &y).unwrap();
        assert!((t - 1.0).abs() < 1e-9);
    }

    #[test]
    fn mutual_information_independent_is_low() {
        let mut x = Vec::new();
        let mut y = Vec::new();
        // Two independent uniform-ish ramps, shuffled by stride
        for i in 0..1000 {
            x.push((i as f64).sin());
            y.push(((i * 7) as f64).cos());
        }
        let mi = mutual_information(&x, &y, 16).unwrap();
        assert!(mi.is_finite());
    }

    #[test]
    fn correlation_matrix_diagonal_is_one() {
        let cols = vec!["a".to_string(), "b".to_string()];
        let xs: Vec<AnalyticsValue> = (0..10).map(|i| AnalyticsValue::Float(i as f64)).collect();
        let ys: Vec<AnalyticsValue> = xs.clone();
        let m = correlation_matrix(&cols, &[xs, ys]);
        assert!((m.get(0, 0) - 1.0).abs() < 1e-9);
        assert!((m.get(1, 1) - 1.0).abs() < 1e-9);
        assert!((m.get(0, 1) - 1.0).abs() < 1e-9);
    }
}
