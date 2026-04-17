//! Competitive rating functions.
//!
//! ELO, Glicko-2, TrueSkill, Bayesian average, Wilson score.

use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// ELO rating system
// ---------------------------------------------------------------------------

/// Returns the expected win probability for player A against player B.
/// Based on the logistic curve: E(A) = 1 / (1 + 10^((Rb - Ra) / 400))
pub fn elo_expected(rating_a: f64, rating_b: f64) -> f64 {
    1.0 / (1.0 + 10.0_f64.powf((rating_b - rating_a) / 400.0))
}

/// Computes the new ELO rating after a match.
/// actual: 1.0 for win, 0.5 for draw, 0.0 for loss.
/// k_factor: typical values are 32 (new players), 24 (established), 16 (masters).
pub fn elo_update(rating: f64, expected: f64, actual: f64, k_factor: f64) -> f64 {
    rating + k_factor * (actual - expected)
}

// ---------------------------------------------------------------------------
// Glicko-2 rating system
// ---------------------------------------------------------------------------

/// Updates a player's Glicko-2 rating after a rating period.
/// rating: current rating (Glicko-2 scale, typically 1500 center)
/// rd: rating deviation (uncertainty, decreases with games played)
/// volatility: expected fluctuation (sigma, typically 0.06)
/// opponents: slice of (opponent_rating, opponent_rd, score) where score is 0.0/0.5/1.0
/// Returns (new_rating, new_rd, new_volatility).
pub fn glicko2_update(
    rating: f64,
    rd: f64,
    volatility: f64,
    opponents: &[(f64, f64, f64)],
) -> Result<(f64, f64, f64)> {
    if opponents.is_empty() {
        // No games: RD increases over time (RD decay)
        let new_rd = (rd * rd + volatility * volatility).sqrt().min(350.0);
        return Ok((rating, new_rd, volatility));
    }

    // Convert to Glicko-2 internal scale
    let mu = (rating - 1500.0) / 173.7178;
    let phi = rd / 173.7178;

    // Step 3: Compute g(phi) and E(mu, mu_j, phi_j) for each opponent
    let mut v_inv = 0.0;
    let mut delta_sum = 0.0;

    for &(opp_rating, opp_rd, score) in opponents {
        let mu_j = (opp_rating - 1500.0) / 173.7178;
        let phi_j = opp_rd / 173.7178;

        let g_j = g_function(phi_j);
        let e_j = e_function(mu, g_j, mu_j);

        v_inv += g_j * g_j * e_j * (1.0 - e_j);
        delta_sum += g_j * (score - e_j);
    }

    if v_inv.abs() < 1e-15 {
        return Ok((rating, rd, volatility));
    }

    let v = 1.0 / v_inv;
    let delta = v * delta_sum;

    // Step 4: Determine new volatility (simplified Illinois algorithm)
    let new_sigma = compute_new_volatility(phi, v, delta, volatility);

    // Step 5: Update phi
    let phi_star = (phi * phi + new_sigma * new_sigma).sqrt();

    // Step 6: Update phi and mu
    let new_phi = 1.0 / (1.0 / (phi_star * phi_star) + 1.0 / v).sqrt();
    let new_mu = mu + new_phi * new_phi * delta_sum;

    // Convert back to Glicko-2 rating scale
    let new_rating = new_mu * 173.7178 + 1500.0;
    let new_rd = (new_phi * 173.7178).min(350.0);

    Ok((new_rating, new_rd, new_sigma))
}

fn g_function(phi: f64) -> f64 {
    1.0 / (1.0 + 3.0 * phi * phi / (std::f64::consts::PI * std::f64::consts::PI)).sqrt()
}

fn e_function(mu: f64, g: f64, mu_j: f64) -> f64 {
    1.0 / (1.0 + (-g * (mu - mu_j)).exp())
}

fn compute_new_volatility(phi: f64, v: f64, delta: f64, sigma: f64) -> f64 {
    let tau = 0.5; // system constant
    let a = (sigma * sigma).ln();
    let epsilon = 0.000001;

    let f = |x: f64| -> f64 {
        let ex = x.exp();
        let d2 = delta * delta;
        let p2 = phi * phi;
        let num = ex * (d2 - p2 - v - ex);
        let den = 2.0 * (p2 + v + ex).powi(2);
        num / den - (x - a) / (tau * tau)
    };

    // Initial bracket
    let mut big_a = a;
    let mut big_b = if delta * delta > phi * phi + v {
        (delta * delta - phi * phi - v).ln()
    } else {
        let mut k = 1.0;
        while f(a - k * tau) < 0.0 {
            k += 1.0;
            if k > 100.0 {
                break;
            }
        }
        a - k * tau
    };

    // Illinois algorithm
    let mut fa = f(big_a);
    let mut fb = f(big_b);

    for _ in 0..100 {
        if (big_b - big_a).abs() <= epsilon {
            break;
        }
        let big_c = big_a + (big_a - big_b) * fa / (fb - fa);
        let fc = f(big_c);

        if fc * fb <= 0.0 {
            big_a = big_b;
            fa = fb;
        } else {
            fa /= 2.0;
        }
        big_b = big_c;
        fb = fc;
    }

    (big_a / 2.0).exp()
}

// ---------------------------------------------------------------------------
// TrueSkill
// ---------------------------------------------------------------------------

/// Simplified TrueSkill update for two teams (1v1).
/// team_ratings: slice of (mu, sigma) pairs.
/// ranks: slice of rank values (0 = 1st place, 1 = 2nd place, etc.).
/// Returns updated (mu, sigma) pairs.
///
/// Uses the simplified two-player formula. For multi-team support,
/// processes teams pairwise in rank order.
pub fn trueskill_update(team_ratings: &[(f64, f64)], ranks: &[u32]) -> Result<Vec<(f64, f64)>> {
    if team_ratings.len() != ranks.len() {
        return Err(ZyronError::ExecutionError(
            "team_ratings and ranks must have the same length".into(),
        ));
    }
    if team_ratings.is_empty() {
        return Ok(Vec::new());
    }
    if team_ratings.len() == 1 {
        return Ok(team_ratings.to_vec());
    }

    let beta = 25.0 / 6.0; // performance variation
    let tau = 25.0 / 300.0; // dynamics factor
    let beta_sq = beta * beta;

    let mut results: Vec<(f64, f64)> = team_ratings.to_vec();

    // Process adjacent pairs in rank order
    let mut indices: Vec<usize> = (0..ranks.len()).collect();
    indices.sort_by_key(|&i| ranks[i]);

    for window in indices.windows(2) {
        let wi = window[0];
        let li = window[1];

        let (mu_w, sigma_w) = results[wi];
        let (mu_l, sigma_l) = results[li];

        // Add dynamics factor
        let sigma_w_sq = sigma_w * sigma_w + tau * tau;
        let sigma_l_sq = sigma_l * sigma_l + tau * tau;

        let c = (2.0 * beta_sq + sigma_w_sq + sigma_l_sq).sqrt();
        let t = (mu_w - mu_l) / c;

        // Normal distribution PDF and CDF approximations
        let v_val = v_function(t);
        let w_val = w_function(t, v_val);

        let mu_w_new = mu_w + sigma_w_sq / c * v_val;
        let mu_l_new = mu_l - sigma_l_sq / c * v_val;

        let sigma_w_new = (sigma_w_sq * (1.0 - sigma_w_sq / (c * c) * w_val))
            .max(0.0001)
            .sqrt();
        let sigma_l_new = (sigma_l_sq * (1.0 - sigma_l_sq / (c * c) * w_val))
            .max(0.0001)
            .sqrt();

        results[wi] = (mu_w_new, sigma_w_new);
        results[li] = (mu_l_new, sigma_l_new);
    }

    Ok(results)
}

fn v_function(t: f64) -> f64 {
    // V(t) = pdf(t) / cdf(t) (truncated Gaussian correction)
    let pdf = normal_pdf(t);
    let cdf = normal_cdf(t);
    if cdf < 1e-15 {
        return -t;
    }
    pdf / cdf
}

fn w_function(t: f64, v: f64) -> f64 {
    v * (v + t)
}

fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Approximation of the standard normal CDF using the Abramowitz and Stegun formula.
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327; // 1/sqrt(2*pi)
    let p = d
        * (-0.5 * x * x).exp()
        * (t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274)))));
    if x >= 0.0 { 1.0 - p } else { p }
}

// ---------------------------------------------------------------------------
// Bayesian average
// ---------------------------------------------------------------------------

/// Computes the Bayesian average (IMDB-style weighted rating).
/// item_avg: average rating for this item.
/// item_count: number of ratings for this item.
/// global_avg: average rating across all items.
/// min_votes: minimum number of votes before the item's own average dominates.
pub fn bayesian_average(item_avg: f64, item_count: u64, global_avg: f64, min_votes: u64) -> f64 {
    let v = item_count as f64;
    let m = min_votes as f64;
    (v * item_avg + m * global_avg) / (v + m)
}

// ---------------------------------------------------------------------------
// Win rate
// ---------------------------------------------------------------------------

/// Simple win rate calculation.
pub fn win_rate(wins: u64, total: u64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    wins as f64 / total as f64
}

// ---------------------------------------------------------------------------
// Wilson score interval
// ---------------------------------------------------------------------------

/// Computes the Wilson score interval lower bound.
/// Used for ranking items by proportion of positive ratings.
/// positive: number of positive ratings.
/// total: total number of ratings.
/// confidence: confidence level (e.g., 0.95 for 95%).
pub fn wilson_score(positive: u64, total: u64, confidence: f64) -> Result<f64> {
    if total == 0 {
        return Ok(0.0);
    }
    if confidence <= 0.0 || confidence >= 1.0 {
        return Err(ZyronError::ExecutionError(format!(
            "Confidence must be between 0 and 1 (exclusive), got {}",
            confidence
        )));
    }

    let n = total as f64;
    let p = positive as f64 / n;

    // z-score for the given confidence level (two-tailed)
    let z = quantile_normal((1.0 + confidence) / 2.0);

    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = p + z2 / (2.0 * n);
    let spread = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();

    Ok((center - spread) / denominator)
}

/// Approximation of the inverse normal CDF (quantile function)
/// using the rational approximation by Peter Acklam.
fn quantile_normal(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Coefficients for the rational approximation
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ELO
    #[test]
    fn test_elo_expected_equal() {
        let e = elo_expected(1500.0, 1500.0);
        assert!((e - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_elo_expected_higher_rated() {
        let e = elo_expected(1800.0, 1500.0);
        assert!(e > 0.5);
        assert!(e < 1.0);
    }

    #[test]
    fn test_elo_expected_lower_rated() {
        let e = elo_expected(1200.0, 1500.0);
        assert!(e < 0.5);
        assert!(e > 0.0);
    }

    #[test]
    fn test_elo_expected_symmetric() {
        let e_a = elo_expected(1500.0, 1800.0);
        let e_b = elo_expected(1800.0, 1500.0);
        assert!((e_a + e_b - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_elo_update_win() {
        let expected = elo_expected(1500.0, 1500.0);
        let new_rating = elo_update(1500.0, expected, 1.0, 32.0);
        assert!(new_rating > 1500.0);
        assert!((new_rating - 1516.0).abs() < 1.0);
    }

    #[test]
    fn test_elo_update_loss() {
        let expected = elo_expected(1500.0, 1500.0);
        let new_rating = elo_update(1500.0, expected, 0.0, 32.0);
        assert!(new_rating < 1500.0);
    }

    #[test]
    fn test_elo_update_draw() {
        let expected = elo_expected(1500.0, 1500.0);
        let new_rating = elo_update(1500.0, expected, 0.5, 32.0);
        assert!((new_rating - 1500.0).abs() < 1e-10);
    }

    // Glicko-2
    #[test]
    fn test_glicko2_no_opponents() {
        let (r, rd, v) = glicko2_update(1500.0, 200.0, 0.06, &[]).unwrap();
        assert_eq!(r, 1500.0);
        assert!(rd > 200.0); // RD increases without games
        assert_eq!(v, 0.06);
    }

    #[test]
    fn test_glicko2_win() {
        let opponents = vec![(1400.0, 30.0, 1.0)]; // beat a 1400 player
        let (r, rd, _) = glicko2_update(1500.0, 200.0, 0.06, &opponents).unwrap();
        assert!(r > 1500.0); // Rating should increase
        assert!(rd < 200.0); // RD should decrease (more certain)
    }

    #[test]
    fn test_glicko2_loss() {
        let opponents = vec![(1400.0, 30.0, 0.0)]; // lost to a 1400 player
        let (r, _rd, _) = glicko2_update(1500.0, 200.0, 0.06, &opponents).unwrap();
        assert!(r < 1500.0);
    }

    #[test]
    fn test_glicko2_multiple_opponents() {
        let opponents = vec![
            (1400.0, 30.0, 1.0),
            (1550.0, 100.0, 0.0),
            (1700.0, 300.0, 0.0),
        ];
        let result = glicko2_update(1500.0, 200.0, 0.06, &opponents);
        assert!(result.is_ok());
        let (r, rd, _) = result.unwrap();
        assert!(rd < 200.0); // RD decreases with more games
        assert!(r.is_finite());
    }

    // TrueSkill
    #[test]
    fn test_trueskill_two_players() {
        let ratings = vec![(25.0, 8.333), (25.0, 8.333)];
        let ranks = vec![0, 1]; // first player won
        let result = trueskill_update(&ratings, &ranks).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result[0].0 > 25.0); // winner's mu increases
        assert!(result[1].0 < 25.0); // loser's mu decreases
    }

    #[test]
    fn test_trueskill_empty() {
        let result = trueskill_update(&[], &[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_trueskill_single() {
        let result = trueskill_update(&[(25.0, 8.0)], &[0]).unwrap();
        assert_eq!(result, vec![(25.0, 8.0)]);
    }

    #[test]
    fn test_trueskill_mismatched_lengths() {
        assert!(trueskill_update(&[(25.0, 8.0)], &[0, 1]).is_err());
    }

    #[test]
    fn test_trueskill_sigma_decreases() {
        let ratings = vec![(25.0, 8.333), (25.0, 8.333)];
        let ranks = vec![0, 1];
        let result = trueskill_update(&ratings, &ranks).unwrap();
        // Both players' uncertainty should decrease
        assert!(result[0].1 < 8.333);
        assert!(result[1].1 < 8.333);
    }

    // Bayesian average
    #[test]
    fn test_bayesian_average_no_votes() {
        let avg = bayesian_average(0.0, 0, 3.5, 10);
        assert_eq!(avg, 3.5); // Falls back to global average
    }

    #[test]
    fn test_bayesian_average_many_votes() {
        let avg = bayesian_average(4.5, 10000, 3.5, 10);
        // With 10000 votes, the item's own average dominates
        assert!((avg - 4.5).abs() < 0.01);
    }

    #[test]
    fn test_bayesian_average_few_votes() {
        let avg = bayesian_average(5.0, 1, 3.5, 10);
        // With 1 vote vs min_votes=10, global average dominates
        assert!(avg < 4.0);
        assert!(avg > 3.5);
    }

    // Win rate
    #[test]
    fn test_win_rate_zero_total() {
        assert_eq!(win_rate(0, 0), 0.0);
    }

    #[test]
    fn test_win_rate_all_wins() {
        assert_eq!(win_rate(10, 10), 1.0);
    }

    #[test]
    fn test_win_rate_half() {
        assert!((win_rate(5, 10) - 0.5).abs() < 1e-10);
    }

    // Wilson score
    #[test]
    fn test_wilson_score_zero() {
        assert_eq!(wilson_score(0, 0, 0.95).unwrap(), 0.0);
    }

    #[test]
    fn test_wilson_score_all_positive() {
        let score = wilson_score(100, 100, 0.95).unwrap();
        assert!(score > 0.9);
        assert!(score < 1.0);
    }

    #[test]
    fn test_wilson_score_mixed() {
        let score = wilson_score(70, 100, 0.95).unwrap();
        assert!(score > 0.5);
        assert!(score < 0.7);
    }

    #[test]
    fn test_wilson_score_more_votes_higher() {
        let score_low = wilson_score(7, 10, 0.95).unwrap();
        let score_high = wilson_score(700, 1000, 0.95).unwrap();
        // Same proportion, more votes -> higher Wilson score (more confidence)
        assert!(score_high > score_low);
    }

    #[test]
    fn test_wilson_score_invalid_confidence() {
        assert!(wilson_score(1, 1, 0.0).is_err());
        assert!(wilson_score(1, 1, 1.0).is_err());
        assert!(wilson_score(1, 1, -0.5).is_err());
    }
}
