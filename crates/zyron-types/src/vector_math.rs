//! Vector space math over f32 slices
//!
//! Inputs mirror the VECTOR column payload, packed f32 values. All
//! accumulation runs in f64 so long vectors do not lose precision, results
//! that produce a new vector narrow back to f32

use zyron_common::{Result, ZyronError};

/// Rejects two vectors of different dimension
fn check_same_len(name: &str, v1: &[f32], v2: &[f32]) -> Result<()> {
    if v1.len() != v2.len() {
        return Err(ZyronError::ExecutionError(format!(
            "{}: vector dimensions differ, {} vs {}",
            name,
            v1.len(),
            v2.len()
        )));
    }
    Ok(())
}

/// Computes the dot product of two equal length vectors
pub fn vector_dot(v1: &[f32], v2: &[f32]) -> Result<f64> {
    check_same_len("vector_dot", v1, v2)?;
    Ok(v1
        .iter()
        .zip(v2.iter())
        .map(|(&a, &b)| a as f64 * b as f64)
        .sum())
}

/// Computes the cross product of two 3 dimensional vectors
pub fn vector_cross(v1: &[f32], v2: &[f32]) -> Result<Vec<f32>> {
    if v1.len() != 3 || v2.len() != 3 {
        return Err(ZyronError::ExecutionError(format!(
            "vector_cross: both vectors must be 3 dimensional, got {} and {}",
            v1.len(),
            v2.len()
        )));
    }
    let (a0, a1, a2) = (v1[0] as f64, v1[1] as f64, v1[2] as f64);
    let (b0, b1, b2) = (v2[0] as f64, v2[1] as f64, v2[2] as f64);
    Ok(vec![
        (a1 * b2 - a2 * b1) as f32,
        (a2 * b0 - a0 * b2) as f32,
        (a0 * b1 - a1 * b0) as f32,
    ])
}

/// Computes the L2 norm of a vector
pub fn vector_norm(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt()
}

/// Scales a vector to unit length. The zero vector has no direction so it
/// is rejected
pub fn vector_normalize(v: &[f32]) -> Result<Vec<f32>> {
    let norm = vector_norm(v);
    if norm == 0.0 {
        return Err(ZyronError::InvalidParameter {
            name: "vector".to_string(),
            value: "zero vector cannot be normalized".to_string(),
        });
    }
    Ok(v.iter().map(|&x| (x as f64 / norm) as f32).collect())
}

/// Computes the angle between two vectors in radians. The cosine is clamped
/// to [-1, 1] before acos so rounding never produces NaN
pub fn vector_angle(v1: &[f32], v2: &[f32]) -> Result<f64> {
    check_same_len("vector_angle", v1, v2)?;
    let n1 = vector_norm(v1);
    let n2 = vector_norm(v2);
    if n1 == 0.0 || n2 == 0.0 {
        return Err(ZyronError::InvalidParameter {
            name: "vector".to_string(),
            value: "zero vector has no angle".to_string(),
        });
    }
    let dot = vector_dot(v1, v2)?;
    Ok((dot / (n1 * n2)).clamp(-1.0, 1.0).acos())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_known_value() {
        let d = vector_dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(d, 32.0);
    }

    #[test]
    fn test_dot_length_mismatch() {
        assert!(vector_dot(&[1.0, 2.0], &[1.0]).is_err());
    }

    #[test]
    fn test_cross_basis_vectors() {
        let c = vector_cross(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]).unwrap();
        assert_eq!(c, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_cross_anticommutative() {
        let a = [2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0];
        let ab = vector_cross(&a, &b).unwrap();
        let ba = vector_cross(&b, &a).unwrap();
        for i in 0..3 {
            assert_eq!(ab[i], -ba[i]);
        }
    }

    #[test]
    fn test_cross_requires_3d() {
        assert!(vector_cross(&[1.0, 0.0], &[0.0, 1.0, 0.0]).is_err());
        assert!(vector_cross(&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0]).is_err());
    }

    #[test]
    fn test_norm() {
        assert_eq!(vector_norm(&[3.0, 4.0]), 5.0);
        assert_eq!(vector_norm(&[]), 0.0);
    }

    #[test]
    fn test_normalize_unit_length() {
        let n = vector_normalize(&[3.0, 4.0]).unwrap();
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
        let norm: f64 = n.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector_errors() {
        let err = vector_normalize(&[0.0, 0.0, 0.0]);
        assert!(matches!(err, Err(ZyronError::InvalidParameter { .. })));
    }

    #[test]
    fn test_angle_orthogonal() {
        let a = vector_angle(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
        assert!((a - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    }

    #[test]
    fn test_angle_parallel_clamps() {
        // identical directions can push the cosine past 1.0 by rounding
        let a = vector_angle(&[1.0, 1.0, 1.0], &[2.0, 2.0, 2.0]).unwrap();
        assert!(a.abs() < 1e-6);
        let opposite = vector_angle(&[1.0, 0.0], &[-1.0, 0.0]).unwrap();
        assert!((opposite - std::f64::consts::PI).abs() < 1e-9);
    }

    #[test]
    fn test_angle_zero_vector_errors() {
        assert!(vector_angle(&[0.0, 0.0], &[1.0, 0.0]).is_err());
    }
}
