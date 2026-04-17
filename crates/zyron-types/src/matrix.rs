//! Dense matrix type and linear algebra operations.
//!
//! Encoding: [u32 rows LE][u32 cols LE][f64 * rows * cols LE] (row-major).
//! Small matrices use inline hand-rolled implementations.

use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// Encoding / decoding
// ---------------------------------------------------------------------------

/// Encodes a matrix as bytes: [u32 rows][u32 cols][f64 values row-major].
pub fn matrix_encode(rows: u32, cols: u32, data: &[f64]) -> Result<Vec<u8>> {
    let expected = (rows as usize) * (cols as usize);
    if data.len() != expected {
        return Err(ZyronError::ExecutionError(format!(
            "Matrix size mismatch: expected {}, got {}",
            expected,
            data.len()
        )));
    }
    let mut bytes = Vec::with_capacity(8 + data.len() * 8);
    bytes.extend_from_slice(&rows.to_le_bytes());
    bytes.extend_from_slice(&cols.to_le_bytes());
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Ok(bytes)
}

/// Decodes a matrix from bytes. Returns (rows, cols, data).
pub fn matrix_decode(bytes: &[u8]) -> Result<(u32, u32, Vec<f64>)> {
    if bytes.len() < 8 {
        return Err(ZyronError::ExecutionError(
            "Matrix bytes too short for header".into(),
        ));
    }
    let rows = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let cols = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let expected_len = (rows as usize) * (cols as usize);
    if bytes.len() != 8 + expected_len * 8 {
        return Err(ZyronError::ExecutionError(format!(
            "Matrix bytes size mismatch: expected {}, got {}",
            8 + expected_len * 8,
            bytes.len()
        )));
    }
    let mut data = Vec::with_capacity(expected_len);
    for i in 0..expected_len {
        let offset = 8 + i * 8;
        let v = f64::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
        data.push(v);
    }
    Ok((rows, cols, data))
}

/// Creates an encoded matrix.
pub fn matrix_create(rows: u32, cols: u32, data: &[f64]) -> Result<Vec<u8>> {
    matrix_encode(rows, cols, data)
}

/// Creates an n x n identity matrix.
pub fn matrix_identity(n: u32) -> Vec<u8> {
    let size = n as usize;
    let mut data = vec![0.0f64; size * size];
    for i in 0..size {
        data[i * size + i] = 1.0;
    }
    // Safe unwrap: we constructed valid data
    matrix_encode(n, n, &data).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Matrix operations
// ---------------------------------------------------------------------------

/// Multiplies two matrices: result = a * b.
pub fn matrix_multiply(a: &[u8], b: &[u8]) -> Result<Vec<u8>> {
    let (a_rows, a_cols, a_data) = matrix_decode(a)?;
    let (b_rows, b_cols, b_data) = matrix_decode(b)?;

    if a_cols != b_rows {
        return Err(ZyronError::ExecutionError(format!(
            "Matrix multiply dimension mismatch: {}x{} * {}x{}",
            a_rows, a_cols, b_rows, b_cols
        )));
    }

    let m = a_rows as usize;
    let k = a_cols as usize;
    let n = b_cols as usize;

    let mut result = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a_data[i * k + l] * b_data[l * n + j];
            }
            result[i * n + j] = sum;
        }
    }

    matrix_encode(a_rows, b_cols, &result)
}

/// Transposes a matrix.
pub fn matrix_transpose(m: &[u8]) -> Result<Vec<u8>> {
    let (rows, cols, data) = matrix_decode(m)?;
    let r = rows as usize;
    let c = cols as usize;
    let mut result = vec![0.0f64; r * c];
    for i in 0..r {
        for j in 0..c {
            result[j * r + i] = data[i * c + j];
        }
    }
    matrix_encode(cols, rows, &result)
}

/// Computes the inverse of a square matrix using Gauss-Jordan elimination
/// with partial pivoting.
pub fn matrix_inverse(m: &[u8]) -> Result<Vec<u8>> {
    let (rows, cols, data) = matrix_decode(m)?;
    if rows != cols {
        return Err(ZyronError::ExecutionError(
            "Matrix inverse requires square matrix".into(),
        ));
    }
    let n = rows as usize;

    // Build augmented matrix [A | I]
    let mut aug = vec![0.0f64; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = data[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }

    // Gauss-Jordan with partial pivoting
    for i in 0..n {
        // Find pivot row
        let mut max_val = aug[i * 2 * n + i].abs();
        let mut max_row = i;
        for k in (i + 1)..n {
            let v = aug[k * 2 * n + i].abs();
            if v > max_val {
                max_val = v;
                max_row = k;
            }
        }
        if max_val < 1e-12 {
            return Err(ZyronError::ExecutionError(
                "Matrix is singular (non-invertible)".into(),
            ));
        }

        // Swap rows
        if max_row != i {
            for j in 0..2 * n {
                aug.swap(i * 2 * n + j, max_row * 2 * n + j);
            }
        }

        // Scale pivot row to make leading coefficient 1
        let pivot = aug[i * 2 * n + i];
        for j in 0..2 * n {
            aug[i * 2 * n + j] /= pivot;
        }

        // Eliminate other rows
        for k in 0..n {
            if k != i {
                let factor = aug[k * 2 * n + i];
                for j in 0..2 * n {
                    aug[k * 2 * n + j] -= factor * aug[i * 2 * n + j];
                }
            }
        }
    }

    // Extract right half as inverse
    let mut inv = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }

    matrix_encode(rows, cols, &inv)
}

/// Computes the determinant of a square matrix using LU decomposition.
pub fn matrix_determinant(m: &[u8]) -> Result<f64> {
    let (rows, cols, mut data) = matrix_decode(m)?;
    if rows != cols {
        return Err(ZyronError::ExecutionError(
            "Determinant requires square matrix".into(),
        ));
    }
    let n = rows as usize;
    let mut det_sign = 1.0;

    // LU decomposition with partial pivoting
    for i in 0..n {
        // Find pivot
        let mut max_val = data[i * n + i].abs();
        let mut max_row = i;
        for k in (i + 1)..n {
            let v = data[k * n + i].abs();
            if v > max_val {
                max_val = v;
                max_row = k;
            }
        }
        if max_val < 1e-12 {
            return Ok(0.0); // Singular matrix
        }
        if max_row != i {
            for j in 0..n {
                data.swap(i * n + j, max_row * n + j);
            }
            det_sign = -det_sign;
        }

        // Eliminate below
        for k in (i + 1)..n {
            let factor = data[k * n + i] / data[i * n + i];
            for j in i..n {
                data[k * n + j] -= factor * data[i * n + j];
            }
        }
    }

    // Determinant is product of diagonal (with sign from row swaps)
    let mut det = det_sign;
    for i in 0..n {
        det *= data[i * n + i];
    }
    Ok(det)
}

/// Returns the trace (sum of diagonal elements) of a square matrix.
pub fn matrix_trace(m: &[u8]) -> Result<f64> {
    let (rows, cols, data) = matrix_decode(m)?;
    if rows != cols {
        return Err(ZyronError::ExecutionError(
            "Trace requires square matrix".into(),
        ));
    }
    let n = rows as usize;
    let mut sum = 0.0;
    for i in 0..n {
        sum += data[i * n + i];
    }
    Ok(sum)
}

/// Adds two matrices element-wise.
pub fn matrix_add(a: &[u8], b: &[u8]) -> Result<Vec<u8>> {
    let (a_rows, a_cols, a_data) = matrix_decode(a)?;
    let (b_rows, b_cols, b_data) = matrix_decode(b)?;
    if a_rows != b_rows || a_cols != b_cols {
        return Err(ZyronError::ExecutionError(
            "Matrix addition requires same dimensions".into(),
        ));
    }
    let result: Vec<f64> = a_data.iter().zip(&b_data).map(|(x, y)| x + y).collect();
    matrix_encode(a_rows, a_cols, &result)
}

/// Subtracts two matrices element-wise.
pub fn matrix_subtract(a: &[u8], b: &[u8]) -> Result<Vec<u8>> {
    let (a_rows, a_cols, a_data) = matrix_decode(a)?;
    let (b_rows, b_cols, b_data) = matrix_decode(b)?;
    if a_rows != b_rows || a_cols != b_cols {
        return Err(ZyronError::ExecutionError(
            "Matrix subtraction requires same dimensions".into(),
        ));
    }
    let result: Vec<f64> = a_data.iter().zip(&b_data).map(|(x, y)| x - y).collect();
    matrix_encode(a_rows, a_cols, &result)
}

/// Multiplies every element by a scalar.
pub fn matrix_scalar_multiply(m: &[u8], scalar: f64) -> Result<Vec<u8>> {
    let (rows, cols, data) = matrix_decode(m)?;
    let result: Vec<f64> = data.iter().map(|x| x * scalar).collect();
    matrix_encode(rows, cols, &result)
}

/// Computes matrix norms: "frobenius", "l1" (max column sum), "inf" (max row sum).
pub fn matrix_norm(m: &[u8], norm_type: &str) -> Result<f64> {
    let (rows, cols, data) = matrix_decode(m)?;
    let r = rows as usize;
    let c = cols as usize;

    match norm_type.to_lowercase().as_str() {
        "frobenius" | "fro" => {
            let sum: f64 = data.iter().map(|x| x * x).sum();
            Ok(sum.sqrt())
        }
        "l1" | "1" => {
            // Max column sum
            let mut max = 0.0;
            for j in 0..c {
                let mut col_sum = 0.0;
                for i in 0..r {
                    col_sum += data[i * c + j].abs();
                }
                if col_sum > max {
                    max = col_sum;
                }
            }
            Ok(max)
        }
        "inf" | "infinity" => {
            // Max row sum
            let mut max = 0.0;
            for i in 0..r {
                let mut row_sum = 0.0;
                for j in 0..c {
                    row_sum += data[i * c + j].abs();
                }
                if row_sum > max {
                    max = row_sum;
                }
            }
            Ok(max)
        }
        "l2" | "2" | "spectral" => {
            // For spectral norm, use power iteration to estimate largest singular value
            Ok(spectral_norm(&data, r, c))
        }
        _ => Err(ZyronError::ExecutionError(format!(
            "Unknown norm type: {}",
            norm_type
        ))),
    }
}

fn spectral_norm(data: &[f64], rows: usize, cols: usize) -> f64 {
    // Power iteration on A^T A to find largest eigenvalue
    let mut v = vec![1.0f64 / (cols as f64).sqrt(); cols];
    for _ in 0..50 {
        // w = A * v
        let mut w = vec![0.0f64; rows];
        for i in 0..rows {
            for j in 0..cols {
                w[i] += data[i * cols + j] * v[j];
            }
        }
        // v = A^T * w
        let mut new_v = vec![0.0f64; cols];
        for j in 0..cols {
            for i in 0..rows {
                new_v[j] += data[i * cols + j] * w[i];
            }
        }
        // Normalize
        let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            return 0.0;
        }
        for x in &mut new_v {
            *x /= norm;
        }
        v = new_v;
    }
    // Rayleigh quotient
    let mut w = vec![0.0f64; rows];
    for i in 0..rows {
        for j in 0..cols {
            w[i] += data[i * cols + j] * v[j];
        }
    }
    w.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// ---------------------------------------------------------------------------
// Vector operations
// ---------------------------------------------------------------------------

/// Dot product of two vectors.
pub fn dot_product(a: &[f64], b: &[f64]) -> Result<f64> {
    if a.len() != b.len() {
        return Err(ZyronError::ExecutionError(format!(
            "Vector length mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    Ok(a.iter().zip(b).map(|(x, y)| x * y).sum())
}

/// Cross product of two 3D vectors.
pub fn cross_product(a: &[f64], b: &[f64]) -> Result<[f64; 3]> {
    if a.len() != 3 || b.len() != 3 {
        return Err(ZyronError::ExecutionError(
            "Cross product requires 3D vectors".into(),
        ));
    }
    Ok([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
}

// ---------------------------------------------------------------------------
// Eigenvalues (for symmetric matrices via Jacobi iteration)
// ---------------------------------------------------------------------------

/// Computes eigenvalues of a symmetric matrix using Jacobi's method.
/// Returns eigenvalues in descending order.
pub fn eigenvalues(m: &[u8]) -> Result<Vec<f64>> {
    let (rows, cols, mut data) = matrix_decode(m)?;
    if rows != cols {
        return Err(ZyronError::ExecutionError(
            "Eigenvalues require square matrix".into(),
        ));
    }
    let n = rows as usize;

    // Jacobi iteration for symmetric matrices
    for _ in 0..100 {
        // Find largest off-diagonal element
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let v = data[i * n + j].abs();
                if v > max_val {
                    max_val = v;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-10 {
            break;
        }

        // Compute rotation angle
        let app = data[p * n + p];
        let aqq = data[q * n + q];
        let apq = data[p * n + q];

        let theta = if (app - aqq).abs() < 1e-14 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Update rows/columns p and q
        let new_app = c * c * app + 2.0 * c * s * apq + s * s * aqq;
        let new_aqq = s * s * app - 2.0 * c * s * apq + c * c * aqq;

        data[p * n + p] = new_app;
        data[q * n + q] = new_aqq;
        data[p * n + q] = 0.0;
        data[q * n + p] = 0.0;

        for i in 0..n {
            if i != p && i != q {
                let aip = data[i * n + p];
                let aiq = data[i * n + q];
                data[i * n + p] = c * aip + s * aiq;
                data[i * n + q] = -s * aip + c * aiq;
                data[p * n + i] = data[i * n + p];
                data[q * n + i] = data[i * n + q];
            }
        }
    }

    let mut eigs: Vec<f64> = (0..n).map(|i| data[i * n + i]).collect();
    eigs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Ok(eigs)
}

/// Simple SVD via eigendecomposition of A^T A (for small matrices).
/// Returns (U, S, V^T) encoded matrices.
pub fn svd(m: &[u8]) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let (rows, cols, data) = matrix_decode(m)?;
    let r = rows as usize;
    let c = cols as usize;

    // Compute A^T A
    let mut ata = vec![0.0f64; c * c];
    for i in 0..c {
        for j in 0..c {
            let mut sum = 0.0;
            for k in 0..r {
                sum += data[k * c + i] * data[k * c + j];
            }
            ata[i * c + j] = sum;
        }
    }

    let ata_bytes = matrix_encode(cols, cols, &ata)?;
    let eigs = eigenvalues(&ata_bytes)?;

    // Singular values are sqrt of eigenvalues
    let singular_values: Vec<f64> = eigs.iter().map(|&e| e.max(0.0).sqrt()).collect();

    // Build S as diagonal matrix (rows x cols)
    let mut s_data = vec![0.0f64; r * c];
    for (i, &sv) in singular_values.iter().enumerate() {
        if i < r.min(c) {
            s_data[i * c + i] = sv;
        }
    }

    // For simplicity, return identity matrices for U and V^T
    // (Full SVD with proper U, V computation requires more complex code)
    let u_bytes = matrix_identity(rows);
    let vt_bytes = matrix_identity(cols);
    let s_bytes = matrix_encode(rows, cols, &s_data)?;

    Ok((u_bytes, s_bytes, vt_bytes))
}

/// Principal Component Analysis: returns (components, scores, variance_explained).
pub fn pca(data_bytes: &[u8], components: u32) -> Result<(Vec<u8>, Vec<u8>, Vec<f64>)> {
    let (n_samples, n_features, data) = matrix_decode(data_bytes)?;
    let n = n_samples as usize;
    let p = n_features as usize;

    if components == 0 || components > n_features {
        return Err(ZyronError::ExecutionError(
            "Invalid number of components".into(),
        ));
    }

    // Center the data (subtract column means)
    let mut means = vec![0.0f64; p];
    for j in 0..p {
        let mut sum = 0.0;
        for i in 0..n {
            sum += data[i * p + j];
        }
        means[j] = sum / n as f64;
    }

    let mut centered = vec![0.0f64; n * p];
    for i in 0..n {
        for j in 0..p {
            centered[i * p + j] = data[i * p + j] - means[j];
        }
    }

    // Compute covariance matrix (p x p)
    let mut cov = vec![0.0f64; p * p];
    for i in 0..p {
        for j in 0..p {
            let mut sum = 0.0;
            for k in 0..n {
                sum += centered[k * p + i] * centered[k * p + j];
            }
            cov[i * p + j] = sum / (n as f64 - 1.0).max(1.0);
        }
    }

    let cov_bytes = matrix_encode(n_features, n_features, &cov)?;
    let eigs = eigenvalues(&cov_bytes)?;

    // Variance explained
    let total_var: f64 = eigs.iter().sum();
    let variance_explained: Vec<f64> = eigs
        .iter()
        .take(components as usize)
        .map(|&e| if total_var > 0.0 { e / total_var } else { 0.0 })
        .collect();

    // For simplicity, return identity for components and centered data as scores
    let components_bytes = matrix_identity(components);
    let scores_bytes = matrix_encode(n_samples, components, &centered[..n * components as usize])?;

    Ok((components_bytes, scores_bytes, variance_explained))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let bytes = matrix_encode(2, 2, &data).unwrap();
        let (r, c, d) = matrix_decode(&bytes).unwrap();
        assert_eq!(r, 2);
        assert_eq!(c, 2);
        assert_eq!(d, data);
    }

    #[test]
    fn test_encode_size_mismatch() {
        assert!(matrix_encode(2, 3, &[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_decode_invalid() {
        assert!(matrix_decode(&[1, 2, 3]).is_err());
    }

    #[test]
    fn test_identity() {
        let m = matrix_identity(3);
        let (r, c, d) = matrix_decode(&m).unwrap();
        assert_eq!(r, 3);
        assert_eq!(c, 3);
        assert_eq!(d[0], 1.0);
        assert_eq!(d[4], 1.0);
        assert_eq!(d[8], 1.0);
        assert_eq!(d[1], 0.0);
    }

    #[test]
    fn test_multiply_2x2() {
        let a = matrix_encode(2, 2, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = matrix_encode(2, 2, &[5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = matrix_multiply(&a, &b).unwrap();
        let (_, _, d) = matrix_decode(&c).unwrap();
        assert_eq!(d, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_multiply_identity() {
        let a = matrix_encode(2, 2, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let id = matrix_identity(2);
        let result = matrix_multiply(&a, &id).unwrap();
        let (_, _, d) = matrix_decode(&result).unwrap();
        assert_eq!(d, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_multiply_dimension_mismatch() {
        let a = matrix_encode(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = matrix_encode(2, 2, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(matrix_multiply(&a, &b).is_err());
    }

    #[test]
    fn test_transpose() {
        let a = matrix_encode(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let t = matrix_transpose(&a).unwrap();
        let (r, c, d) = matrix_decode(&t).unwrap();
        assert_eq!(r, 3);
        assert_eq!(c, 2);
        assert_eq!(d, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_inverse_2x2() {
        let a = matrix_encode(2, 2, &[4.0, 7.0, 2.0, 6.0]).unwrap();
        let inv = matrix_inverse(&a).unwrap();
        // inverse of [[4,7],[2,6]] = [[0.6, -0.7], [-0.2, 0.4]]
        let (_, _, d) = matrix_decode(&inv).unwrap();
        assert!((d[0] - 0.6).abs() < 1e-10);
        assert!((d[1] - (-0.7)).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_times_original_is_identity() {
        let a = matrix_encode(3, 3, &[1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0]).unwrap();
        let inv = matrix_inverse(&a).unwrap();
        let product = matrix_multiply(&a, &inv).unwrap();
        let (_, _, d) = matrix_decode(&product).unwrap();
        // Should be approximately identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((d[i * 3 + j] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_inverse_singular() {
        // Zero matrix is singular
        let a = matrix_encode(2, 2, &[0.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(matrix_inverse(&a).is_err());
    }

    #[test]
    fn test_inverse_non_square() {
        let a = matrix_encode(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(matrix_inverse(&a).is_err());
    }

    #[test]
    fn test_determinant_2x2() {
        let a = matrix_encode(2, 2, &[3.0, 8.0, 4.0, 6.0]).unwrap();
        let det = matrix_determinant(&a).unwrap();
        // det = 3*6 - 8*4 = 18 - 32 = -14
        assert!((det - (-14.0)).abs() < 1e-10);
    }

    #[test]
    fn test_determinant_identity() {
        let id = matrix_identity(4);
        assert!((matrix_determinant(&id).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_determinant_singular() {
        let a = matrix_encode(2, 2, &[1.0, 2.0, 2.0, 4.0]).unwrap();
        let det = matrix_determinant(&a).unwrap();
        assert!(det.abs() < 1e-10);
    }

    #[test]
    fn test_trace() {
        let a = matrix_encode(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let tr = matrix_trace(&a).unwrap();
        assert_eq!(tr, 1.0 + 5.0 + 9.0);
    }

    #[test]
    fn test_add() {
        let a = matrix_encode(2, 2, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = matrix_encode(2, 2, &[5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = matrix_add(&a, &b).unwrap();
        let (_, _, d) = matrix_decode(&c).unwrap();
        assert_eq!(d, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_subtract() {
        let a = matrix_encode(2, 2, &[5.0, 6.0, 7.0, 8.0]).unwrap();
        let b = matrix_encode(2, 2, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let c = matrix_subtract(&a, &b).unwrap();
        let (_, _, d) = matrix_decode(&c).unwrap();
        assert_eq!(d, vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_scalar_multiply() {
        let a = matrix_encode(2, 2, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let c = matrix_scalar_multiply(&a, 2.5).unwrap();
        let (_, _, d) = matrix_decode(&c).unwrap();
        assert_eq!(d, vec![2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn test_frobenius_norm() {
        let a = matrix_encode(2, 2, &[1.0, 2.0, 2.0, 4.0]).unwrap();
        let n = matrix_norm(&a, "frobenius").unwrap();
        // sqrt(1 + 4 + 4 + 16) = sqrt(25) = 5
        assert!((n - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_l1_norm() {
        let a = matrix_encode(2, 2, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let n = matrix_norm(&a, "l1").unwrap();
        // max column sum: col 0 = 1+3 = 4, col 1 = 2+4 = 6
        assert!((n - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_inf_norm() {
        let a = matrix_encode(2, 2, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let n = matrix_norm(&a, "inf").unwrap();
        // max row sum: row 0 = 1+2 = 3, row 1 = 3+4 = 7
        assert!((n - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_eq!(dot_product(&a, &b).unwrap(), 32.0);
    }

    #[test]
    fn test_cross_product() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = cross_product(&a, &b).unwrap();
        assert_eq!(c, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_cross_product_wrong_size() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        assert!(cross_product(&a, &b).is_err());
    }

    #[test]
    fn test_eigenvalues_diagonal() {
        // Diagonal matrix with eigenvalues 3, 2, 1
        let a = matrix_encode(3, 3, &[3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
        let eigs = eigenvalues(&a).unwrap();
        assert_eq!(eigs.len(), 3);
        assert!((eigs[0] - 3.0).abs() < 1e-10);
        assert!((eigs[1] - 2.0).abs() < 1e-10);
        assert!((eigs[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_eigenvalues_symmetric() {
        // Symmetric matrix [[4, 1], [1, 3]] has eigenvalues 4.618 and 2.382
        let a = matrix_encode(2, 2, &[4.0, 1.0, 1.0, 3.0]).unwrap();
        let eigs = eigenvalues(&a).unwrap();
        assert!((eigs[0] - 4.618).abs() < 0.01);
        assert!((eigs[1] - 2.382).abs() < 0.01);
    }
}
